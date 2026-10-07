#!/usr/bin/env python3
import cv2
import json
import os
import argparse
import sys
import threading
import csv
import rosbag2_py
import multiprocessing
import math
import numpy as np
import re
from bisect import bisect_left
from collections import defaultdict
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message
from rosidl_runtime_py.convert import message_to_ordereddict
from cv_bridge import CvBridge
from .common import get_current_timestamp_iso8601
import h5py


def parse_stage_timestamp(value):
    """Return the current video_tags stage timestamp object fields."""
    if not isinstance(value, dict):
        return None, None
    return value.get("cpu_ts"), value.get("frame_relative")

def _valid_timestamp(value):
    return isinstance(value, int) and value >= 0

def explicit_annotation_timestamp_ns(frame):
    """Return an explicit CPU realtime timestamp for annotation/extraction."""
    realtime = frame.get("cpu", {}).get("realtime", {})
    if not isinstance(realtime, dict):
        return None

    mono = realtime.get("mono_source_ns")
    if _valid_timestamp(mono):
        return mono

    left = realtime.get("left_source_ns")
    right = realtime.get("right_source_ns")
    has_left = _valid_timestamp(left)
    has_right = _valid_timestamp(right)
    if has_left and has_right:
        return left // 2 + right // 2 + ((left % 2) + (right % 2)) // 2
    if has_left:
        return left
    if has_right:
        return right

    recorder = realtime.get("recorder_reception_ns")
    if _valid_timestamp(recorder):
        return recorder
    return None

def flatten_dict(d, parent_key='', sep='.'):
    items = []
    if isinstance(d, (dict,)) or (hasattr(d, 'items') and callable(getattr(d, 'items'))):
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.extend(flatten_dict(v, new_key, sep=sep).items())
    elif isinstance(d, (list, tuple, np.ndarray)):
        for i, v in enumerate(d):
            new_key = f"{parent_key}{sep}{i}"
            items.extend(flatten_dict(v, new_key, sep=sep).items())
    else:
        items.append((parent_key, d))
    return dict(items)

def natural_sort_key(s):
    """
    Key function for natural sorting (e.g., 'frame10' comes after 'frame2').
    """
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-123456789]+)', str(s))]

def detect_rosbag_format(bag_path):
    if os.path.isdir(bag_path):
        mcap_files = [f for f in os.listdir(bag_path) if f.endswith('.mcap')]
        if mcap_files: return 'mcap'
        db3_files = [f for f in os.listdir(bag_path) if f.endswith('.db3')]
        if db3_files: return 'sqlite3'
    
    reader = rosbag2_py.SequentialReader()
    for storage_id in ['mcap', 'sqlite3']:
        try:
            reader.open(rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
                        rosbag2_py.ConverterOptions('', ''))
            return storage_id
        except Exception: continue
    return None

def _message_time_seconds(msg):
    """Return a ROS message timestamp as floating-point seconds."""
    header = getattr(msg, "header", None)
    if header is not None:
        stamp = getattr(header, "stamp", None)
        if stamp is not None:
            sec = getattr(stamp, "sec", 0)
            nanosec = getattr(stamp, "nanosec", 0)
            return float(sec) + float(nanosec) * 1e-9
    return None


def _joint_positions(msg, count):
    positions = np.asarray(msg.position, dtype=np.float32)
    if len(positions) >= count:
        return positions[:count]

    effort = np.asarray(msg.effort, dtype=np.float32)
    if len(effort) >= count:
        return effort[:count]

    return np.zeros(count, dtype=np.float32)


def _joint_state(msg, count):
    positions = _joint_positions(msg, count)
    velocities = np.asarray(msg.velocity, dtype=np.float32)
    velocities = velocities[:count] if len(velocities) >= count else np.pad(
        velocities, (0, count - len(velocities)), constant_values=0.0
    )
    efforts = np.asarray(msg.effort, dtype=np.float32)
    efforts = efforts[:count] if len(efforts) >= count else np.pad(
        efforts, (0, count - len(efforts)), constant_values=0.0
    )
    return positions, velocities, efforts


def _read_rosbag_messages(bag_path, start_ns=None, end_ns=None):
    storage_id = detect_rosbag_format(bag_path)
    if not storage_id:
        raise ValueError(f"Unable to detect rosbag format for {bag_path}")

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=bag_path, storage_id=storage_id),
        rosbag2_py.ConverterOptions("", ""),
    )

    topic_types = {
        topic.name: topic.type
        for topic in reader.get_all_topics_and_types()
    }

    while reader.has_next():
        topic, data, timestamp = reader.read_next()
        if start_ns is not None and timestamp < start_ns:
            continue
        if end_ns is not None and timestamp > end_ns:
            continue
        msg_type = get_message(topic_types[topic])
        yield topic, deserialize_message(data, msg_type), timestamp


def _nearest_indexed(timestamps, values, target_t, tolerance):
    """Return the value nearest target_t, provided it is within tolerance."""
    if not len(timestamps):
        return None

    if target_t <= timestamps[0]:
        index = 0
    elif target_t >= timestamps[-1]:
        index = len(timestamps) - 1
    else:
        index = bisect_left(timestamps, target_t)
        if index == len(timestamps):
            index = len(timestamps) - 1
        elif index > 0:
            left_dt = abs(timestamps[index - 1] - target_t)
            right_dt = abs(timestamps[index] - target_t)
            if right_dt < left_dt:
                index = index
            else:
                index -= 1

    best_t = timestamps[index]
    if abs(best_t - target_t) > tolerance:
        return None
    return values[index]


def _latest_before_or_nearest_indexed(timestamps, values, target_t, tolerance):
    """Return the latest value at or before target_t if it is sufficiently recent."""
    if not len(timestamps):
        return None

    index = bisect_left(timestamps, target_t)
    if index > 0:
        latest_index = index - 1
        latest_t = timestamps[latest_index]
        if abs(latest_t - target_t) <= tolerance:
            return values[latest_index]

    nearest = _nearest_indexed(timestamps, values, target_t, tolerance)
    if nearest is not None:
        return nearest
    return None


def _nearest_from_sequence(samples, target_t, tolerance):
    if not samples:
        return None
    timestamps = np.asarray([ts for ts, _ in samples], dtype=np.float64)
    values = [value for _, value in samples]
    return _nearest_indexed(timestamps, values, target_t, tolerance)


def _latest_before_or_nearest(samples, target_t, tolerance):
    if not samples:
        return None
    timestamps = np.asarray([ts for ts, _ in samples], dtype=np.float64)
    values = [value for _, value in samples]
    return _latest_before_or_nearest_indexed(timestamps, values, target_t, tolerance)


def _image_to_array(msg, encoding="bgr8"):
    bridge = CvBridge()
    return bridge.imgmsg_to_cv2(msg, desired_encoding=encoding)


def _select_nearest_frame_indices(frame_timestamps, target_timestamps, tolerance):
    """Return the nearest sidecar frame index for each target timestamp."""
    selected = []
    valid_indices = [
        index for index, timestamp in enumerate(frame_timestamps)
        if timestamp is not None
    ]
    valid_timestamps = np.asarray(
        [frame_timestamps[index] for index in valid_indices],
        dtype=np.float64,
    )
    valid_values = np.asarray(valid_indices, dtype=np.int64)

    for target_t in target_timestamps:
        index = _nearest_indexed(valid_timestamps, valid_values, target_t, tolerance)
        if index is None:
            candidate_index = int(np.argmin(np.abs(valid_timestamps - target_t)))
            index = valid_values[candidate_index]
        selected.append(int(index))
    return selected


def _read_video_images(video_path, frame_indices):
    if not os.path.exists(video_path):
        return {"left": [], "right": []}

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video file {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        if width <= 0:
            raise RuntimeError(f"Video has invalid width: {video_path}")
        half_width = width // 2

        images = {"left": [], "right": []}
        current_pos = None
        for frame_index in frame_indices:
            if current_pos is None or frame_index < current_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
                current_pos = frame_index
            while current_pos < frame_index:
                if not cap.grab():
                    break
                current_pos += 1
            if current_pos != frame_index:
                break
            success, frame = cap.read()
            if not success:
                break
            left = frame[:, :half_width].copy()
            right = frame[:, half_width:].copy()
            images["left"].append(left)
            images["right"].append(right)
            current_pos += 1
        return images
    finally:
        cap.release()


def write_episode_hdf5(path, qpos, qvel, action, images, timestamps,
                      source_times, arm_effort, spatial_cf,
                      measured_cf=None, joint_force=None, metadata=None):
    """Write an ACT-compatible HDF5 episode."""
    with h5py.File(path, "w") as f:
        metadata = metadata or {}
        for key, value in metadata.items():
            if key == "camera_names":
                f.attrs[key] = np.asarray(value, dtype="S")
            elif value is None:
                f.attrs[key] = "null"
            elif isinstance(value, (dict, list, tuple)):
                f.attrs[key] = json.dumps(value)
            else:
                f.attrs[key] = value

        f.attrs["sim"] = False
        f.attrs["source"] = "dvrk"
        f.attrs["color_order"] = metadata.get("image_encoding", "bgr8")
        f.attrs["state_semantics"] = "PSM_arm[:arm_joints] | PSM_jaw[:jaw_joints]"
        f.attrs["action_semantics"] = "cmd_arm[:arm_joints] | cmd_jaw[:jaw_joints]"
        f.attrs["qvel_semantics"] = (
            "PSM_arm_vel[:arm_joints] | PSM_jaw_vel[:jaw_joints]"
        )

        obs = f.create_group("observations")
        obs.create_dataset("qpos", data=np.asarray(qpos, dtype=np.float32))
        obs.create_dataset("qvel", data=np.asarray(qvel, dtype=np.float32))

        img_group = obs.create_group("images")
        for name, frames in images.items():
            img_group.create_dataset(
                name,
                data=np.stack(frames),
                dtype="u1",
                compression="gzip",
                compression_opts=2,
            )

        f.create_dataset("action", data=np.asarray(action, dtype=np.float32))
        f.create_dataset("timestamp", data=np.asarray(timestamps, dtype=np.float64))

        source_group = f.create_group("source_timestamps")
        for name, values in source_times.items():
            source_group.create_dataset(name, data=np.asarray(values, dtype=np.float64))

        f.create_dataset("arm_effort", data=np.asarray(arm_effort, dtype=np.float32))
        f.create_dataset("spatial_cf", data=np.asarray(spatial_cf, dtype=np.float32))

        if measured_cf is not None:
            f.create_dataset("measured_cf", data=np.asarray(measured_cf, dtype=np.float32))
        if joint_force is not None:
            f.create_dataset("joint_force", data=np.asarray(joint_force, dtype=np.float32))


def _find_sidecar_video(source_directory, sidecar_video=None):
    if sidecar_video and os.path.exists(sidecar_video):
        return sidecar_video
    if not source_directory:
        return None

    base_name = os.path.splitext(os.path.basename(sidecar_video or ""))[0]
    if not base_name:
        return None
    sidecar_path = os.path.join(source_directory, f"{base_name}.json")
    if os.path.exists(sidecar_path):
        return sidecar_path
    return None


def rosbag_to_act_hdf5(bag_path, output_path, source_directory=None,
                      fps=10.0, state_tolerance=0.03,
                      image_tolerance=0.08, command_tolerance=0.05,
                      arm_joints=6, jaw_joints=1, resize=None,
                      sidecar_video=None):
    """Convert a dVRK ROS bag into a synchronized ACT episode HDF5 file."""
    required = {
        "/PSM2/measured_js",
        "/PSM2/jaw/measured_js",
        "/PSM2/setpoint_js",
        "/PSM2/jaw/setpoint_js",
        "/PSM2/spatial/measured_cf",
    }

    topic_messages = defaultdict(list)
    for topic, msg, bag_ts in _read_rosbag_messages(bag_path):
        if topic in required or topic in {
            "/PSM2/body/measured_cf",
            "/PSM2/gravity_compensation/setpoint_js",
            "/measured_cf",
            "/wrist/cam1/image_raw",
        }:
            extracted = _message_time_seconds(msg)
            if extracted is None:
                extracted = float(bag_ts) * 1e-9
            topic_messages[topic].append((extracted, msg))

    if not topic_messages.get("/PSM2/measured_js"):
        raise ValueError(f"No PSM state messages found in {bag_path}")

    topic_arrays = {
        topic: (
            np.asarray([ts for ts, _ in samples], dtype=np.float64),
            [value for _, value in samples],
        )
        for topic, samples in topic_messages.items()
    }

    joint_dim = arm_joints + jaw_joints
    qpos_list, qvel_list, action_list = [], [], []
    timestamps, arm_effort_list, spatial_cf_list = [], [], []
    measured_cf_list, joint_force_list = [], []
    source_times = {
        "psm_arm": [], "psm_jaw": [], "cmd_arm": [], "cmd_jaw": [],
        "left": [], "right": [], "wrist_1": [], "wrist_2": [],
    }
    images = {"left": [], "right": [], "wrist_1": []}
    dt = 1.0 / fps

    sidecar_path = _find_sidecar_video(source_directory, sidecar_video)
    sidecar_data = None
    sidecar_video_path = None
    sidecar_frame_timestamps = []
    if sidecar_path is not None:
        sidecar_video_path = sidecar_path
        if sidecar_video_path.endswith(".json"):
            with open(sidecar_video_path, "r") as f:
                sidecar_data = json.load(f)
            video_file = sidecar_data.get("video_file")
            if video_file:
                sidecar_video_path = os.path.join(source_directory, video_file)
            sidecar_frame_timestamps = []
            for frame in sidecar_data.get("frames", []):
                timestamp_ns = explicit_annotation_timestamp_ns(frame)
                if timestamp_ns is None:
                    sidecar_frame_timestamps.append(None)
                else:
                    sidecar_frame_timestamps.append(float(timestamp_ns) / 1e9)

    arm_times, arm_messages = topic_arrays["/PSM2/measured_js"]
    start_t = float(arm_times[0])
    end_t = float(arm_times[-1])
    sample_count = int(math.ceil((end_t - start_t) / dt)) + 1

    for i in range(sample_count):
        target_t = start_t + i * dt
        arm_msg = _nearest_indexed(arm_times, arm_messages, target_t, state_tolerance)
        jaw_times, jaw_messages = topic_arrays["/PSM2/jaw/measured_js"]
        jaw_msg = _nearest_indexed(jaw_times, jaw_messages, target_t, state_tolerance)
        if arm_msg is None or jaw_msg is None:
            continue

        arm_ts = _nearest_indexed(arm_times, arm_times, target_t, state_tolerance)
        jaw_ts = _nearest_indexed(jaw_times, jaw_times, target_t, state_tolerance)
        if arm_ts is None or jaw_ts is None:
            continue

        arm_pos, arm_vel, arm_effort = _joint_state(arm_msg, arm_joints)
        jaw_pos, jaw_vel, jaw_effort = _joint_state(jaw_msg, jaw_joints)

        cmd_arm_times, cmd_arm_messages = topic_arrays["/PSM2/setpoint_js"]
        cmd_jaw_times, cmd_jaw_messages = topic_arrays["/PSM2/jaw/setpoint_js"]
        cmd_arm = _latest_before_or_nearest_indexed(cmd_arm_times, cmd_arm_messages, arm_ts, command_tolerance)
        cmd_jaw = _latest_before_or_nearest_indexed(cmd_jaw_times, cmd_jaw_messages, arm_ts, command_tolerance)
        if cmd_arm is None or cmd_jaw is None:
            continue

        qpos = np.concatenate([arm_pos, jaw_pos], axis=0)
        qvel = np.concatenate([arm_vel, jaw_vel], axis=0)
        action = np.concatenate([
            _joint_positions(cmd_arm, arm_joints),
            _joint_positions(cmd_jaw, jaw_joints),
        ], axis=0)

        qpos_list.append(qpos)
        qvel_list.append(qvel)
        action_list.append(action)
        timestamps.append(float(arm_ts))
        arm_effort_list.append(arm_effort)

        spatial_times, spatial_messages = topic_arrays["/PSM2/spatial/measured_cf"]
        spatial_cf = _nearest_indexed(spatial_times, spatial_messages, arm_ts, state_tolerance)
        if spatial_cf is not None:
            spatial_cf_list.append(np.array([
                spatial_cf.wrench.force.x,
                spatial_cf.wrench.force.y,
                spatial_cf.wrench.force.z,
                spatial_cf.wrench.torque.x,
                spatial_cf.wrench.torque.y,
                spatial_cf.wrench.torque.z,
            ], dtype=np.float32))
        else:
            spatial_cf_list.append(np.zeros(6, dtype=np.float32))

        measured_times, measured_messages = topic_arrays.get("/measured_cf", (np.asarray([]), []))
        measured_cf = _nearest_indexed(measured_times, measured_messages, arm_ts, state_tolerance)
        if measured_cf is not None:
            measured_cf_list.append(np.array([
                measured_cf.wrench.force.x,
                measured_cf.wrench.force.y,
                measured_cf.wrench.force.z,
                measured_cf.wrench.torque.x,
                measured_cf.wrench.torque.y,
                measured_cf.wrench.torque.z,
            ], dtype=np.float32))

        gravity_times, gravity_messages = topic_arrays.get(
            "/PSM2/gravity_compensation/setpoint_js",
            (np.asarray([]), []),
        )
        gravity = _nearest_indexed(gravity_times, gravity_messages, arm_ts, state_tolerance)
        if gravity is not None:
            joint_force_list.append(np.asarray(_joint_positions(gravity, arm_joints), dtype=np.float32))

        source_times["psm_arm"].append(float(arm_ts))
        source_times["psm_jaw"].append(float(jaw_ts))
        source_times["cmd_arm"].append(float(_message_time_seconds(cmd_arm)))
        source_times["cmd_jaw"].append(float(_message_time_seconds(cmd_jaw)))

    if not qpos_list:
        raise ValueError("No synchronized samples were found in the ROS bag")

    if sidecar_video_path is not None and os.path.exists(sidecar_video_path):
        selected_indices = _select_nearest_frame_indices(
            sidecar_frame_timestamps,
            timestamps,
            image_tolerance,
        )
        valid_indices = [idx for idx in selected_indices if idx is not None]
        if valid_indices:
            video_frames = _read_video_images(sidecar_video_path, valid_indices)
            images["left"] = [video_frames["left"][i] for i in range(len(valid_indices))]
            images["right"] = [video_frames["right"][i] for i in range(len(valid_indices))]
            source_times["left"] = [
                float(sidecar_frame_timestamps[idx])
                for idx in valid_indices
            ]
            source_times["right"] = source_times["left"]

    wrist_key = "/wrist/cam1/image_raw"
    wrist_times, wrist_messages = topic_arrays.get(wrist_key, (np.asarray([]), []))
    if len(wrist_times):
        wrist_images = []
        wrist_times_out = []
        for target_t in timestamps:
            sample = _nearest_indexed(wrist_times, wrist_messages, target_t, image_tolerance)
            if sample is None:
                continue
            wrist_images.append(_image_to_array(sample, "bgr8"))
            wrist_times_out.append(float(target_t))
        images["wrist_1"] = wrist_images
        source_times["wrist_1"] = wrist_times_out

    if len(images.get("left", [])) != len(qpos_list):
        raise ValueError(
            f"Stereo video contains {len(images.get('left', []))} frames but "
            f"{len(qpos_list)} synchronized samples were produced"
        )

    if len(images.get("right", [])) != len(qpos_list):
        raise ValueError(
            f"Stereo video right frames ({len(images.get('right', []))}) do not match "
            f"{len(qpos_list)} samples"
        )

    metadata = {
        "source": "dvrk",
        "sim": False,
        "fps": float(fps),
        "dt": 1.0 / float(fps),
        "requested_steps": int(len(qpos_list)),
        "captured_steps": int(len(qpos_list)),
        "dropped_steps": 0,
        "capture_ratio": 1.0,
        "state_tolerance_sec": float(state_tolerance),
        "image_tolerance_sec": float(image_tolerance),
        "command_tolerance_sec": float(command_tolerance),
        "joint_dim": int(joint_dim),
        "wall_start_time": float(start_t),
        "wall_end_time": float(end_t),
        "recorded_at_iso": get_current_timestamp_iso8601(),
        "command_mode": "latest_before_or_nearest",
        "camera_names": ["left", "right", "wrist_1"],
        "image_encoding": "bgr8",
        "topic_names": {
            "psm_arm_topic": "/PSM2/measured_js",
            "psm_jaw_topic": "/PSM2/jaw/measured_js",
            "cmd_arm_topic": "/PSM2/setpoint_js",
            "cmd_jaw_topic": "/PSM2/jaw/setpoint_js",
            "left_topic": "/stereo/left/image_raw",
            "right_topic": "/stereo/right/image_raw",
            "wrist1_topic": "/wrist/cam1/image_raw",
            "wrist2_topic": "none",
        },
        "resize_w": 0,
        "resize_h": 0,
        "resize": resize,
    }

    write_episode_hdf5(
        output_path,
        qpos=np.stack(qpos_list),
        qvel=np.stack(qvel_list),
        action=np.stack(action_list),
        images={
            "left": images["left"],
            "right": images["right"],
            "wrist_1": [],
        },
        timestamps=np.asarray(timestamps, dtype=np.float64),
        source_times=source_times,
        arm_effort=np.stack(arm_effort_list),
        spatial_cf=np.stack(spatial_cf_list),
        measured_cf=np.stack(measured_cf_list) if measured_cf_list else None,
        joint_force=np.stack(joint_force_list) if joint_force_list else None,
        metadata=metadata,
    )

    print(f"Finished ACT HDF5 conversion: {output_path}")


def process_video_chunk(args):
    video_path, output_dir, timestamps, frame_indices, output_format, video_basename, is_ns, is_ms, side_by_side = args
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return 0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    half_width = width // 2

    saved_count = 0
    current_pos = None
    for ts, frame_idx in zip(timestamps, frame_indices):
        if current_pos is None or frame_idx < current_pos or frame_idx - current_pos > 500:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            current_pos = frame_idx
        while current_pos < frame_idx:
            if not cap.grab():
                break
            current_pos += 1
        if current_pos != frame_idx:
            break
        ret, frame = cap.read()
        if not ret: break
        current_pos += 1
        
        if is_ns: ts_str = f"{int(ts) / 1e9:.9f}"
        elif is_ms: ts_str = f"{float(ts) / 1000.0:.6f}"
        else: ts_str = f"{float(ts):.6f}"
        
        if side_by_side in ["", "LR", "RL"]:
            left = frame[:, :half_width]
            right = frame[:, half_width:]
            
            if side_by_side == "RL":
                left, right = right, left
                
            cv2.imwrite(os.path.join(output_dir, f"{video_basename}_left_{ts_str}.{output_format}"), left)
            cv2.imwrite(os.path.join(output_dir, f"{video_basename}_right_{ts_str}.{output_format}"), right)
        else:
            image_name = f"{video_basename}_{ts_str}.{output_format}"
            cv2.imwrite(os.path.join(output_dir, image_name), frame)
        saved_count += 1
    cap.release()
    return saved_count

def extract_video_range(video_path, output_path, start_frame, end_frame, fps, original_indices=None, side_by_side="undefined"):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return False
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # If using OpenCV loop, this fps setting is for the output
    t_fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = t_fps if t_fps > 0 else 30.0

    out = None
    out_l = None
    out_r = None

    if side_by_side in ["LR", "RL"]:
        base_path = os.path.splitext(output_path)[0]
        out_l_path = f"{base_path}_left.mp4"
        out_r_path = f"{base_path}_right.mp4"
        
        half_width = width // 2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        
        print(f"Splitting video into: {out_l_path} and {out_r_path}")
        out_l = cv2.VideoWriter(out_l_path, fourcc, fps, (half_width, height))
        out_r = cv2.VideoWriter(out_r_path, fourcc, fps, (half_width, height))
        
        if not out_l.isOpened() or not out_r.isOpened():
             print("Error: Failed to open output video files.")
             cap.release()
             if out_l: out_l.release()
             if out_r: out_r.release()
             return False

    else:
        # Standard mono output using standard MP4 writer (or GStreamer equivalent)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        if not out.isOpened():
            cap.release()
            return False

    # Process frames
    success = True
    try:
        def process_frame(f):
            if out:
                out.write(f)
            else:
                # side_by_side processing
                curr_w = f.shape[1]
                hw = curr_w // 2
                left_img = f[:, :hw]
                right_img = f[:, hw:]
                
                if side_by_side == "RL":
                     out_l.write(right_img)
                     out_r.write(left_img)
                else:
                     out_l.write(left_img)
                     out_r.write(right_img)

        # Determine which frames to extract
        if original_indices is not None:
            sorted_indices = sorted(list(set(original_indices)))
        else:
            sorted_indices = list(range(start_frame, end_frame + 1))
            
        if not sorted_indices:
            print("No frames to extract.")
            return True

        # Initial seek to the first desired frame
        first_target = sorted_indices[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_target)
        
        # Verify seek: if we are at 0 but wanted something else, assume seek failure
        # Note: CAP_PROP_POS_FRAMES is not always accurate, but 0 usually means reset.
        current_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
        if current_pos == 0 and first_target > 0:
             print(f"Warning: Seek to frame {first_target} failed (pos={current_pos}). Fallback to manual skip.")
             cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
             current_pos = 0
             while current_pos < first_target:
                  if not cap.grab():
                       print(f"Error: Could not skip to frame {first_target}.")
                       break
                  current_pos += 1
        else:
             # Assume seek succeeded, or we are at 0 and wanted 0
             current_pos = first_target

        for target_idx in sorted_indices:
            # If we need to go forward
            if target_idx > current_pos:
                skip = target_idx - current_pos
                # If gap is small, skip. If large, try seek (unless we know seek is broken?)
                # For robustness, let's prefer skipping if it's reasonable (< 500 frames)
                # or if we suspect seek is broken.
                # Here we'll just skip to be safe and consistent.
                for _ in range(skip):
                    if not cap.grab():
                        current_pos = -1 # invalidate
                        break
                    current_pos += 1
            
            # If we somehow need to go back (shouldn't happen with sorted list)
            elif target_idx < current_pos:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                current_pos = target_idx # Assume it works...
            
            if current_pos != target_idx:
                 break
                 
            ret, frame = cap.read()
            if not ret:
                print(f"Error: Could not read frame {target_idx}")
                break
            
            process_frame(frame)
            current_pos += 1
            
    except Exception as e:
        print(f"Error during extraction: {e}")
        success = False
    finally:
        if out: out.release()
        if out_l: out_l.release()
        if out_r: out_r.release()
        cap.release()
    
    return success



def extract_session_data(json_path, output_dir, formats, num_jobs, start_acq=None, end_acq=None, latency_s=0.0, split_stereo=False):
    try:
        with open(json_path, 'r') as f: data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_path}: {e}")
        return

    if data.get("type") != "dvrk_data:video_sidecar@1.0.0":
        print(f"CRITICAL: Unsupported sidecar type in {json_path}: {data.get('type')}")
        return

    video_filename = data.get("video_file")
    video_path = None
    if video_filename:
        video_path = os.path.join(os.path.dirname(json_path), video_filename)
    if not video_path or not os.path.exists(video_path):
        base = os.path.splitext(json_path)[0]
        if os.path.exists(base + ".mp4"):
            video_path = base + ".mp4"

    if not video_path: return

    frames = data.get("frames")
    if not isinstance(frames, list) or not frames:
        print(f"CRITICAL: Sidecar {json_path} is missing current frames[] data")
        return

    timestamps = [explicit_annotation_timestamp_ns(f) for f in frames]
    missing = [i for i, (f, ts) in enumerate(zip(frames, timestamps))
               if ts is None
               or not isinstance(f.get("gst"), dict)
               or f["gst"].get("pts_ns") is None]
    if missing:
        print(f"CRITICAL: Sidecar {json_path} has frames without explicit annotation time/GStreamer PTS")
        return

    is_ns = True
    is_ms = False

    indices = []
    for i, ts in enumerate(timestamps):
        if (start_acq is None or ts >= start_acq) and (end_acq is None or ts <= end_acq):
            indices.append(i)
    
    if not indices: return
    
    video_indices = [int(frames[i].get("frame_index", i)) for i in indices]
    
    start_idx = video_indices[0]
    end_idx = video_indices[-1]
    filtered_timestamps = [timestamps[i] for i in indices]
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    side_by_side = data.get("side_by_side", "undefined")
    if not split_stereo:
        side_by_side = "undefined"
    
    for fmt in formats:
        if fmt == 'mp4':
            fps = data.get("fps") or 30.0
            out_all_name = f"{video_basename}.mp4"
            print(f"Extracting video segment: {out_all_name}")
            extract_video_range(video_path, os.path.join(output_dir, out_all_name), start_idx, end_idx, fps, video_indices, side_by_side)
        else:
            tasks = []
            chunk_size = math.ceil(len(video_indices) / num_jobs)
            for i in range(num_jobs):
                s = i * chunk_size
                e = min((i + 1) * chunk_size, len(video_indices))
                if s >= len(video_indices): break
                tasks.append((video_path, output_dir, [filtered_timestamps[k] for k in range(s, e)], video_indices[s:e], fmt, video_basename, is_ns, is_ms, side_by_side))
            
            if num_jobs > 1:
                with multiprocessing.Pool(processes=num_jobs) as pool: pool.map(process_video_chunk, tasks)
            else:
                process_video_chunk(tasks[0])

def index_videos(index_data):
    if index_data.get("type") != "dvrk_data:index@1.0.0":
        print(f"CRITICAL: Unsupported index type: {index_data.get('type')}")
        return []

    streams = index_data.get("streams")
    if not isinstance(streams, list):
        print("CRITICAL: Current index format requires streams[]")
        return []

    videos = []
    for stream in streams:
        if not isinstance(stream, dict):
            continue
        stream_name = stream.get("name", "")
        for video in stream.get("videos", []) or []:
            if not isinstance(video, dict) or not video.get("file"):
                continue
            item = dict(video)
            if stream_name:
                item["gst_input"] = stream_name
            videos.append(item)

    return videos

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--directory", help="Session directory", required=True)
    parser.add_argument("-t", "--tags", help="Tags JSON file")
    parser.add_argument("-a", "--all", action="store_true", help="Extract all data")
    parser.add_argument("-f", "--format", choices=['jpg', 'png', 'mp4'], action='append', help="Output format(s).")
    parser.add_argument("-j", "--jobs", type=int)
    parser.add_argument("-S", "--split-stereo", action="store_true", help="Split stereo video streams (LR/RL) into two channels")
    parser.add_argument("--hdf5", action="store_true", help="Write ACT-compatible HDF5 episode(s) from the ROS bag")
    parser.add_argument("--fps", type=float, default=10.0, help="ACT episode sampling rate in Hz")
    args = parser.parse_args()

    if not args.tags and not args.all:
        print("Error: Either -t (tags) or -a (all) must be specified.")
        sys.exit(1)

    formats = args.format if args.format else ['mp4']
    num_jobs = args.jobs or max(1, (os.cpu_count() or 1) // 2)
    
    if not os.path.exists(args.directory):
        print(f"CRITICAL: Session directory does not exist: {args.directory}")
        sys.exit(1)

    index_path = os.path.join(args.directory, "index.json")
    if not os.path.exists(index_path):
        print(f"CRITICAL: index.json not found in {args.directory}")
        sys.exit(1)

    with open(index_path, 'r') as f: index_data = json.load(f)
    
    videos = index_videos(index_data)
    rosbag_name = index_data.get("rosbags")

    base_extracted_dir = os.path.join(args.directory, "extracted")
    os.makedirs(base_extracted_dir, exist_ok=True)

    extraction_targets = []
    if args.all:
        extraction_targets.append({"name": "full_session", "start": None, "end": None})
    if args.tags:
        tags_path = args.tags
        if not os.path.exists(tags_path):
            tags_path = os.path.join(args.directory, os.path.basename(args.tags))
            if not os.path.exists(tags_path):
                print(f"CRITICAL: Tag file not found: {args.tags} (also checked {tags_path})")
                sys.exit(1)
        with open(tags_path, 'r') as f: tags_data = json.load(f)
        
        tags_type = tags_data.get("type", "unknown")
        if tags_type not in ["dvrk_data:video_tags@1.0.0", "dvrk_data:session_tags@1.0.0"]:
             print(f"CRITICAL: Unsupported tags type: {tags_type}")
             sys.exit(1)
        
        if tags_type == "dvrk_data:session_tags@1.0.0":
             print("CRITICAL: Session tags are not yet supported for extraction.")
             sys.exit(1)

        source_v_name = tags_data.get("video_file")
        source_v_ts = []
        if source_v_name:
            for v in videos:
                if v["file"] == source_v_name:
                    v_base = os.path.splitext(v["file"])[0]
                    v_json = os.path.join(args.directory, f"{v_base}.json")
                    if os.path.exists(v_json):
                        with open(v_json, 'r') as vf:
                            v_data = json.load(vf)
                            frames = v_data.get("frames", [])
                            source_v_ts = [explicit_annotation_timestamp_ns(f)
                                           for f in frames]
                    break

        stage_counts = {}
        for stage in tags_data.get("stages", []):
            name = stage["name"]
            
            ts_s, f_s = parse_stage_timestamp(stage.get("start"))
            ts_e, f_e = parse_stage_timestamp(stage.get("end"))
            
            def resolve_acq(ts, f):
                # Handle types
                if f is not None:
                    try: f = int(f)
                    except: f = None
                
                if f is not None and source_v_ts:
                    if 0 <= f < len(source_v_ts):
                        final_val = source_v_ts[f]
                        if final_val is not None:
                            return int(final_val)
                
                if ts is not None:
                    return int(ts)

                return None

            start_val = resolve_acq(ts_s, f_s)
            end_val = resolve_acq(ts_e, f_e)
                
            stage_counts[name] = stage_counts.get(name, 0) + 1
            dir_name = f"{name}_{stage_counts[name]:03d}"
            extraction_targets.append({"name": dir_name, "start": start_val, "end": end_val})

    for target in extraction_targets:
        target_dir = os.path.join(base_extracted_dir, target["name"])
        os.makedirs(target_dir, exist_ok=True)
        print(f"Processing target: {target['name']} (start={target['start']}, end={target['end']})")
        
        for v in videos:
            v_base = os.path.splitext(v["file"])[0]
            v_json = os.path.join(args.directory, f"{v_base}.json")
            if os.path.exists(v_json):
                extract_session_data(v_json, target_dir, formats, num_jobs, target["start"], target["end"], split_stereo=args.split_stereo)
        
        if rosbag_name:
            bag_path = os.path.join(args.directory, rosbag_name)
            if os.path.exists(bag_path):
                hdf5_path = os.path.join(target_dir, f"{target['name']}.hdf5")
                if args.hdf5:
                    sidecar_json = None
                    for v in videos:
                        v_base = os.path.splitext(v["file"])[0]
                        candidate = os.path.join(args.directory, f"{v_base}.json")
                        if os.path.exists(candidate):
                            sidecar_json = candidate
                            break
                    rosbag_to_act_hdf5(
                        bag_path,
                        hdf5_path,
                        source_directory=args.directory,
                        fps=args.fps,
                        sidecar_video=sidecar_json,
                    )
                else:
                    raise RuntimeError(
                        "The raw ROS-bag HDF5 converter has been replaced by the ACT episode converter. "
                        "Use --hdf5."
                    )

    print(f"Extraction complete. Files extracted to: {base_extracted_dir}")

if __name__ == "__main__":
    main()
