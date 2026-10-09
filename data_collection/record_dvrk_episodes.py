#!/usr/bin/env python3
# record_dvrk_episodes.py
#
# Core recording backend and CLI for dVRK teleoperation data collection.
# Produces ACT-compatible HDF5 files containing state, commands, force/torque, and cameras.
#
# Key Features:
#   1. High-Speed Asynchronous ROS 2 Intake:
#      Dedicated background executor services callbacks continuously as messages publish.
#   2. Precise Nearest-Timestep Matching:
#      Uses binary search (bisect) to match the closest timesteps of each topic stream.
#   3. Frequency-Controlled Saving:
#      Resamples data onto a fixed timeline grid at the specified rate (--fps).
#   4. Always Collects All Force/Torque Modalities:
#      Spatial force/torque, body force/torque, ATI /measured_cf, and gravity compensation.
#   5. PyQt GUI Integration:
#      The PyQt application is modularized in `record_gui.py` and can be launched directly
#      via `python record_gui.py` or `python record_dvrk_episodes.py --gui`.

import os
import sys
import time
import shutil
import signal
import subprocess
import threading
import argparse
from bisect import bisect_left
from collections import deque
from datetime import datetime

import numpy as np

try:
    import h5py
except ImportError:
    h5py = None

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import termios
    import tty
    import select
except ImportError:
    termios = None
    tty = None
    select = None

# ROS 2 imports
try:
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import Image, JointState
    from geometry_msgs.msg import WrenchStamped
    from cv_bridge import CvBridge
    ROS_AVAILABLE = True
except ImportError:
    rclpy = None
    Node = object
    SingleThreadedExecutor = None
    Image = None
    JointState = None
    WrenchStamped = None
    CvBridge = None
    ROS_AVAILABLE = False


# ============================================================
# Utilities
# ============================================================

def wait_for_key(key='q'):
    """Wait for keypress in terminal (POSIX termios + Windows msvcrt fallback)."""
    print(f'\nPress "{key}" to start recording (Ctrl+C to quit)')
    if termios and tty and select:
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setcbreak(fd)
            while True:
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if not rlist:
                    continue
                ch = sys.stdin.read(1)
                if ch == key:
                    return True
                if ch == '\x03':
                    raise KeyboardInterrupt
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    else:
        try:
            import msvcrt
            while True:
                if msvcrt.kbhit():
                    ch = msvcrt.getch().decode('utf-8', errors='ignore')
                    if ch == key:
                        return True
                    if ch == '\x03':
                        raise KeyboardInterrupt
                time.sleep(0.05)
        except Exception:
            input(f'Press Enter to start recording (target key: "{key}")...')
            return True


def get_next_episode_idx(output_dir):
    """Find the next available episode index in output_dir."""
    existing = []
    if not os.path.exists(output_dir):
        return 0
    for name in os.listdir(output_dir):
        if name.startswith("episode_") and name.endswith(".hdf5"):
            try:
                existing.append(int(name[len("episode_"):-len(".hdf5")]))
            except ValueError:
                pass
    return 0 if not existing else max(existing) + 1


def msg_time_to_sec(msg):
    """Extract float timestamp in seconds from ROS msg header."""
    if msg is not None and hasattr(msg, 'header'):
        header = getattr(msg, 'header', None)
        if header is not None and hasattr(header, 'stamp'):
            stamp = header.stamp
            sec = getattr(stamp, 'sec', None)
            nanosec = getattr(stamp, 'nanosec', None)
            if sec is not None and nanosec is not None:
                t = float(sec) + float(nanosec) * 1e-9
                if t > 0.0:
                    return t
            nanos = getattr(stamp, 'nanoseconds', None)
            if nanos is not None:
                t = float(nanos) * 1e-9
                if t > 0.0:
                    return t
            if isinstance(stamp, (int, float)) and stamp > 0:
                return float(stamp)
    return time.time()


def start_rosbag_recording(bag_path, topics):
    """Start ros2 bag recording subprocess."""
    cmd = ["ros2", "bag", "record", "-o", bag_path] + [t for t in topics if t and t.lower() != 'none']
    print(f"[rosbag] Starting: {' '.join(cmd)}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        preexec_fn=getattr(os, "setsid", None)
    )
    return proc


def stop_rosbag_recording(proc):
    """Terminate ros2 bag recording cleanly."""
    if proc is None:
        return
    try:
        if hasattr(os, "killpg") and hasattr(os, "getpgid"):
            os.killpg(os.getpgid(proc.pid), signal.SIGINT)
        else:
            proc.send_signal(signal.SIGINT)
        proc.wait(timeout=10)
        print("[rosbag] Stopped")
    except Exception as e:
        print(f"[rosbag] Stop error: {e}")
        try:
            proc.kill()
        except Exception:
            pass


def derive_topic_names(psm="PSM2", wrist_cameras=None, left_topic=None, right_topic=None,
                       wrist1_topic=None, wrist2_topic=None):
    """Derives standard dVRK topics based on PSM choice."""
    wrist_cameras = wrist_cameras or []
    return {
        "psm_arm_topic":     f"/{psm}/measured_js",
        "psm_jaw_topic":     f"/{psm}/jaw/measured_js",
        "cmd_arm_topic":     f"/{psm}/setpoint_js",
        "cmd_jaw_topic":     f"/{psm}/jaw/setpoint_js",
        "spatial_cf_topic":  f"/{psm}/spatial/measured_cf",
        "body_cf_topic":     f"/{psm}/body/measured_cf",
        "gravity_topic":     f"/{psm}/gravity_compensation/setpoint_js",
        "measured_cf_topic": "/measured_cf",
        "left_topic":        left_topic or "/console1/left/image_raw",
        "right_topic":       right_topic or "/console1/right/image_raw",
        "wrist1_topic":      (wrist1_topic or "/wrist/cam1/image_raw") if "cam1" in wrist_cameras else "none",
        "wrist2_topic":      (wrist2_topic or "/wrist/cam2/image_raw") if "cam2" in wrist_cameras else "none",
    }


# ============================================================
# Binary-Search Nearest-Timestep Matching
# ============================================================

def find_closest_sample(timestamps, data_list, target_t, tolerance=None):
    """
    Find sample closest to target_t within tolerance using binary search O(log M).
    """
    if not timestamps or not data_list:
        return None, None
    idx = bisect_left(timestamps, target_t)
    candidates = []
    if idx < len(timestamps):
        candidates.append((abs(timestamps[idx] - target_t), idx))
    if idx > 0:
        candidates.append((abs(timestamps[idx - 1] - target_t), idx - 1))

    if not candidates:
        return None, None

    best_dt, best_idx = min(candidates, key=lambda x: x[0])
    if tolerance is not None and best_dt > tolerance:
        return None, None
    return timestamps[best_idx], data_list[best_idx]


def find_latest_before_or_closest(timestamps, data_list, target_t, tolerance=None):
    """
    Find latest sample before target_t within tolerance, falling back to closest.
    """
    if not timestamps or not data_list:
        return None, None
    idx = bisect_left(timestamps, target_t)
    if idx > 0:
        dt = target_t - timestamps[idx - 1]
        if tolerance is None or dt <= tolerance:
            return timestamps[idx - 1], data_list[idx - 1]
    return find_closest_sample(timestamps, data_list, target_t, tolerance)


# ============================================================
# High-Speed ROS 2 Intake Node
# ============================================================

class DVRKRecorder(Node):
    """
    ROS 2 Node that subscribes to all dVRK state, command, force/torque, and camera topics.
    Pulls messages as fast as they are published by the system and stores them into
    per-topic trial buffers during an active recording session.
    """

    def __init__(self,
                 psm="PSM2",
                 wrist_cameras=None,
                 left_topic="/console1/left/image_raw",
                 right_topic="/console1/right/image_raw",
                 wrist1_topic="/wrist/cam1/image_raw",
                 wrist2_topic="/wrist/cam2/image_raw",
                 arm_joints=6,
                 jaw_joints=1,
                 image_encoding='bgr8'):

        super().__init__('dvrk_recorder')

        self.bridge = CvBridge() if CvBridge is not None else None
        self.image_encoding = image_encoding
        self.arm_joints = arm_joints
        self.jaw_joints = jaw_joints
        self.joint_dim = arm_joints + jaw_joints
        self.psm = psm
        self.wrist_cameras = wrist_cameras or ['cam1', 'cam2']

        # Live statistics: topic -> {'count': int, 'last_ts': float, 'rate_hz': float}
        self.topic_stats = {}
        self.stats_lock = threading.Lock()
        self._rate_trackers = {}

        # Latest preview image for GUI
        self.latest_preview_img = None
        self.preview_lock = threading.Lock()

        # Recording session management
        self.is_recording = False
        self.recording_lock = threading.Lock()
        self.trial_buffers = self._create_empty_buffers()
        self.trial_start_wall = 0.0

        self.subscriptions_dict = {}
        self.topic_names = derive_topic_names(
            psm=self.psm,
            wrist_cameras=self.wrist_cameras,
            left_topic=left_topic,
            right_topic=right_topic,
            wrist1_topic=wrist1_topic,
            wrist2_topic=wrist2_topic
        )
        self._init_subscriptions()

        self.get_logger().info(f"DVRKRecorder active. PSM: {self.psm} (joint_dim={self.joint_dim})")

    def _create_empty_buffers(self):
        return {
            "psm_arm":     [],
            "psm_jaw":     [],
            "cmd_arm":     [],
            "cmd_jaw":     [],
            "spatial_cf":  [],
            "body_cf":     [],
            "measured_cf": [],
            "gravity":     [],
            "left":        [],
            "right":       [],
            "wrist_1":     [],
            "wrist_2":     [],
        }

    def _init_subscriptions(self):
        for sub in self.subscriptions_dict.values():
            try:
                self.destroy_subscription(sub)
            except Exception:
                pass
        self.subscriptions_dict.clear()

        # JointState subscribers
        self._add_sub(JointState, self.topic_names["psm_arm_topic"], self.psm_arm_cb, 50)
        self._add_sub(JointState, self.topic_names["psm_jaw_topic"], self.psm_jaw_cb, 50)
        self._add_sub(JointState, self.topic_names["cmd_arm_topic"], self.cmd_arm_cb, 50)
        self._add_sub(JointState, self.topic_names["cmd_jaw_topic"], self.cmd_jaw_cb, 50)

        # Force/torque subscribers (ALWAYS COLLECTED)
        self._add_sub(WrenchStamped, self.topic_names["spatial_cf_topic"], self.spatial_cf_cb, 50)
        self._add_sub(WrenchStamped, self.topic_names["body_cf_topic"], self.body_cf_cb, 50)
        self._add_sub(WrenchStamped, self.topic_names["measured_cf_topic"], self.measured_cf_cb, 50)
        self._add_sub(JointState, self.topic_names["gravity_topic"], self.gravity_cb, 50)

        # Camera subscribers
        self._add_sub(Image, self.topic_names["left_topic"], self.left_cb, 20)
        self._add_sub(Image, self.topic_names["right_topic"], self.right_cb, 20)
        if self.topic_names["wrist1_topic"].lower() != 'none':
            self._add_sub(Image, self.topic_names["wrist1_topic"], self.wrist1_cb, 20)
        if self.topic_names["wrist2_topic"].lower() != 'none':
            self._add_sub(Image, self.topic_names["wrist2_topic"], self.wrist2_cb, 20)

    def _add_sub(self, msg_type, topic, callback, qos):
        if not topic or topic.lower() == 'none':
            return
        sub = self.create_subscription(msg_type, topic, callback, qos)
        self.subscriptions_dict[topic] = sub
        with self.stats_lock:
            if topic not in self.topic_stats:
                self.topic_stats[topic] = {'count': 0, 'last_ts': 0.0, 'rate_hz': 0.0}
                self._rate_trackers[topic] = deque(maxlen=40)

    def update_configuration(self, psm="PSM2", wrist_cameras=None, left_topic=None, right_topic=None):
        """Update PSM or camera subscriptions dynamically."""
        self.psm = psm
        if wrist_cameras is not None:
            self.wrist_cameras = wrist_cameras
        self.topic_names = derive_topic_names(
            psm=self.psm,
            wrist_cameras=self.wrist_cameras,
            left_topic=left_topic or self.topic_names.get("left_topic"),
            right_topic=right_topic or self.topic_names.get("right_topic"),
        )
        self._init_subscriptions()

    def _track_msg(self, topic):
        now = time.time()
        with self.stats_lock:
            stats = self.topic_stats.get(topic)
            if stats is not None:
                stats['count'] += 1
                stats['last_ts'] = now
                dq = self._rate_trackers.get(topic)
                if dq is not None:
                    dq.append(now)
                    if len(dq) > 5:
                        dt = dq[-1] - dq[0]
                        if dt > 0.001:
                            stats['rate_hz'] = (len(dq) - 1) / dt

    def start_recording(self):
        with self.recording_lock:
            self.trial_buffers = self._create_empty_buffers()
            self.trial_start_wall = time.time()
            self.is_recording = True

    def stop_recording(self):
        with self.recording_lock:
            self.is_recording = False
            buffers_copy = self.trial_buffers
            self.trial_buffers = self._create_empty_buffers()
            return buffers_copy, self.trial_start_wall

    def get_live_stats(self):
        with self.stats_lock:
            snapshot = {}
            now = time.time()
            for topic, data in self.topic_stats.items():
                active = (now - data['last_ts']) < 1.0 if data['last_ts'] > 0 else False
                rate = data['rate_hz'] if active else 0.0
                snapshot[topic] = {
                    'count': data['count'],
                    'last_ts': data['last_ts'],
                    'rate_hz': rate,
                    'active': active
                }
            return snapshot

    # ── Callbacks ──────────────────────────────────────────────

    def _js_positions(self, msg, n_joints):
        pos = np.array(msg.position, dtype=np.float32)
        if len(pos) >= n_joints:
            return pos[:n_joints]
        effort = np.array(msg.effort, dtype=np.float32)
        if len(effort) >= n_joints:
            return effort[:n_joints]
        return np.zeros(n_joints, dtype=np.float32)

    def _js_state(self, msg, n_joints):
        pos = self._js_positions(msg, n_joints)
        vel = np.array(msg.velocity, dtype=np.float32) if msg.velocity else np.zeros(n_joints, dtype=np.float32)
        vel = vel[:n_joints] if len(vel) >= n_joints else np.pad(vel, (0, n_joints - len(vel)), 'constant')
        effort = np.array(msg.effort, dtype=np.float32) if msg.effort else np.zeros(n_joints, dtype=np.float32)
        effort = effort[:n_joints] if len(effort) >= n_joints else np.pad(effort, (0, n_joints - len(effort)), 'constant')
        return pos, vel, effort

    def psm_arm_cb(self, msg):
        self._track_msg(self.topic_names["psm_arm_topic"])
        if self.is_recording:
            ts = msg_time_to_sec(msg)
            self.trial_buffers["psm_arm"].append((ts, self._js_state(msg, self.arm_joints)))

    def psm_jaw_cb(self, msg):
        self._track_msg(self.topic_names["psm_jaw_topic"])
        if self.is_recording:
            ts = msg_time_to_sec(msg)
            self.trial_buffers["psm_jaw"].append((ts, self._js_state(msg, self.jaw_joints)))

    def cmd_arm_cb(self, msg):
        self._track_msg(self.topic_names["cmd_arm_topic"])
        if self.is_recording:
            ts = msg_time_to_sec(msg)
            self.trial_buffers["cmd_arm"].append((ts, self._js_positions(msg, self.arm_joints)))

    def cmd_jaw_cb(self, msg):
        self._track_msg(self.topic_names["cmd_jaw_topic"])
        if self.is_recording:
            ts = msg_time_to_sec(msg)
            self.trial_buffers["cmd_jaw"].append((ts, self._js_positions(msg, self.jaw_joints)))

    def spatial_cf_cb(self, msg):
        self._track_msg(self.topic_names["spatial_cf_topic"])
        if self.is_recording:
            ts = msg_time_to_sec(msg)
            wrench = np.array([
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
            ], dtype=np.float32)
            self.trial_buffers["spatial_cf"].append((ts, wrench))

    def body_cf_cb(self, msg):
        self._track_msg(self.topic_names["body_cf_topic"])
        if self.is_recording:
            ts = msg_time_to_sec(msg)
            wrench = np.array([
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
            ], dtype=np.float32)
            self.trial_buffers["body_cf"].append((ts, wrench))

    def measured_cf_cb(self, msg):
        self._track_msg(self.topic_names["measured_cf_topic"])
        if self.is_recording:
            ts = msg_time_to_sec(msg)
            wrench = np.array([
                msg.wrench.force.x, msg.wrench.force.y, msg.wrench.force.z,
                msg.wrench.torque.x, msg.wrench.torque.y, msg.wrench.torque.z
            ], dtype=np.float32)
            self.trial_buffers["measured_cf"].append((ts, wrench))

    def gravity_cb(self, msg):
        self._track_msg(self.topic_names["gravity_topic"])
        if self.is_recording:
            ts = msg_time_to_sec(msg)
            pos = self._js_positions(msg, self.arm_joints)
            self.trial_buffers["gravity"].append((ts, pos))

    def _img_cb(self, msg, stream_name, topic):
        self._track_msg(topic)
        cv_img = None
        if self.bridge is not None:
            try:
                cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding=self.image_encoding)
            except Exception:
                pass

        if stream_name == "left" and cv_img is not None:
            with self.preview_lock:
                self.latest_preview_img = cv_img

        if self.is_recording and cv_img is not None:
            ts = msg_time_to_sec(msg)
            self.trial_buffers[stream_name].append((ts, cv_img))

    def left_cb(self, msg): self._img_cb(msg, "left", self.topic_names["left_topic"])
    def right_cb(self, msg): self._img_cb(msg, "right", self.topic_names["right_topic"])
    def wrist1_cb(self, msg): self._img_cb(msg, "wrist_1", self.topic_names["wrist1_topic"])
    def wrist2_cb(self, msg): self._img_cb(msg, "wrist_2", self.topic_names["wrist2_topic"])


# ============================================================
# Trial Synchronization
# ============================================================

def synchronize_trial_data(trial_buffers,
                           fps=100.0,
                           image_fps=20.0,
                           state_tolerance=0.03,
                           image_tolerance=0.08,
                           command_tolerance=0.05,
                           requested_seconds=None,
                           resize=None,
                           command_mode='latest_before_or_nearest',
                           wrist_cameras=None,
                           wall_start_time=None):
    """
    Matches the closest timesteps of all topics together at the specified frequency (fps),
    extracting state, commands, force/torque, and camera frames into synchronized arrays.
    """
    wrist_cameras = wrist_cameras or []
    dt = 1.0 / float(fps)
    image_interval = max(1, int(round(fps / image_fps)))

    arm_items = trial_buffers.get("psm_arm", [])
    if len(arm_items) < 2:
        return None

    arm_times = [item[0] for item in arm_items]
    arm_vals  = [item[1] for item in arm_items]

    jaw_items = trial_buffers.get("psm_jaw", [])
    jaw_times = [item[0] for item in jaw_items]
    jaw_vals  = [item[1] for item in jaw_items]

    cmd_arm_items = trial_buffers.get("cmd_arm", [])
    cmd_arm_times = [item[0] for item in cmd_arm_items]
    cmd_arm_vals  = [item[1] for item in cmd_arm_items]

    cmd_jaw_items = trial_buffers.get("cmd_jaw", [])
    cmd_jaw_times = [item[0] for item in cmd_jaw_items]
    cmd_jaw_vals  = [item[1] for item in cmd_jaw_items]

    spatial_items = trial_buffers.get("spatial_cf", [])
    spatial_times = [item[0] for item in spatial_items]
    spatial_vals  = [item[1] for item in spatial_items]

    body_items = trial_buffers.get("body_cf", [])
    body_times = [item[0] for item in body_items]
    body_vals  = [item[1] for item in body_items]

    meas_items = trial_buffers.get("measured_cf", [])
    meas_times = [item[0] for item in meas_items]
    meas_vals  = [item[1] for item in meas_items]

    grav_items = trial_buffers.get("gravity", [])
    grav_times = [item[0] for item in grav_items]
    grav_vals  = [item[1] for item in grav_items]

    left_items = trial_buffers.get("left", [])
    left_times = [item[0] for item in left_items]
    left_vals  = [item[1] for item in left_items]

    right_items = trial_buffers.get("right", [])
    right_times = [item[0] for item in right_items]
    right_vals  = [item[1] for item in right_items]

    w1_items = trial_buffers.get("wrist_1", [])
    w1_times = [item[0] for item in w1_items]
    w1_vals  = [item[1] for item in w1_items]

    w2_items = trial_buffers.get("wrist_2", [])
    w2_times = [item[0] for item in w2_items]
    w2_vals  = [item[1] for item in w2_items]

    t_start = arm_times[0]
    if requested_seconds is not None and requested_seconds > 0:
        requested_steps = int(round(requested_seconds * fps))
    else:
        duration = max(0.0, arm_times[-1] - arm_times[0])
        requested_steps = int(round(duration * fps))

    if requested_steps < 2:
        return None

    target_timestamps = [t_start + k * dt for k in range(requested_steps)]

    qpos_list = []
    qvel_list = []
    action_list = []
    timestamps = []
    arm_effort_list = []
    spatial_cf_list = []
    body_cf_list = []
    measured_cf_list = []
    joint_force_list = []

    cam_names = ["left", "right"]
    if "cam1" in wrist_cameras: cam_names.append("wrist_1")
    if "cam2" in wrist_cameras: cam_names.append("wrist_2")

    images = {k: [] for k in cam_names}
    image_timestamps = []
    image_indices = []
    image_source_times = {k: [] for k in cam_names}
    state_source_times = {k: [] for k in ["psm_arm", "psm_jaw", "cmd_arm", "cmd_jaw"]}

    accepted_steps = 0
    dropped_steps = 0

    cmd_matcher = (find_latest_before_or_closest
                   if command_mode == 'latest_before_or_nearest'
                   else find_closest_sample)

    for k, target_t in enumerate(target_timestamps):
        is_image_tick = (k % image_interval == 0)

        t_arm, arm_val = find_closest_sample(arm_times, arm_vals, target_t, state_tolerance)
        if arm_val is None:
            dropped_steps += 1
            continue
        arm_pos, arm_vel, arm_effort = arm_val

        t_jaw, jaw_val = find_closest_sample(jaw_times, jaw_vals, t_arm, state_tolerance)
        if jaw_val is None:
            dropped_steps += 1
            continue
        jaw_pos, jaw_vel, _ = jaw_val

        t_cmd_arm, cmd_arm_pos = cmd_matcher(cmd_arm_times, cmd_arm_vals, t_arm, command_tolerance)
        if cmd_arm_pos is None:
            dropped_steps += 1
            continue

        t_cmd_jaw, cmd_jaw_pos = cmd_matcher(cmd_jaw_times, cmd_jaw_vals, t_arm, command_tolerance)
        if cmd_jaw_pos is None:
            dropped_steps += 1
            continue

        _, spatial_val = find_closest_sample(spatial_times, spatial_vals, t_arm, state_tolerance)
        if spatial_val is None:
            dropped_steps += 1
            continue

        _, body_val = find_closest_sample(body_times, body_vals, t_arm, state_tolerance)
        if body_val is None:
            dropped_steps += 1
            continue

        _, meas_val = find_closest_sample(meas_times, meas_vals, t_arm, state_tolerance)
        if meas_val is None:
            dropped_steps += 1
            continue

        _, grav_pos = find_closest_sample(grav_times, grav_vals, t_arm, state_tolerance)
        if grav_pos is None:
            dropped_steps += 1
            continue

        tick_images = {}
        tick_source_cams = {}
        if is_image_tick:
            t_left, left_frame = find_closest_sample(left_times, left_vals, t_arm, image_tolerance)
            t_right, right_frame = find_closest_sample(right_times, right_vals, t_arm, image_tolerance)
            if left_frame is None or right_frame is None:
                dropped_steps += 1
                continue

            tick_images["left"] = left_frame.copy()
            tick_images["right"] = right_frame.copy()
            tick_source_cams["left"] = float(t_left)
            tick_source_cams["right"] = float(t_right)

            if "cam1" in wrist_cameras:
                t_w1, w1_frame = find_closest_sample(w1_times, w1_vals, t_arm, image_tolerance)
                if w1_frame is None:
                    dropped_steps += 1
                    continue
                tick_images["wrist_1"] = w1_frame.copy()
                tick_source_cams["wrist_1"] = float(t_w1)

            if "cam2" in wrist_cameras:
                t_w2, w2_frame = find_closest_sample(w2_times, w2_vals, t_arm, image_tolerance)
                if w2_frame is None:
                    dropped_steps += 1
                    continue
                tick_images["wrist_2"] = w2_frame.copy()
                tick_source_cams["wrist_2"] = float(t_w2)

            if resize is not None and cv2 is not None:
                w, h = resize
                tick_images = {k: cv2.resize(v, (w, h), interpolation=cv2.INTER_AREA)
                               for k, v in tick_images.items()}

        qpos = np.concatenate([arm_pos, jaw_pos], axis=0)
        qvel = np.concatenate([arm_vel, jaw_vel], axis=0)
        action = np.concatenate([cmd_arm_pos, cmd_jaw_pos], axis=0)
        joint_force = arm_effort - grav_pos

        qpos_list.append(qpos)
        qvel_list.append(qvel)
        action_list.append(action)
        timestamps.append(float(t_arm))
        arm_effort_list.append(arm_effort)
        spatial_cf_list.append(spatial_val)
        body_cf_list.append(body_val)
        measured_cf_list.append(meas_val)
        joint_force_list.append(joint_force)

        state_source_times["psm_arm"].append(float(t_arm))
        state_source_times["psm_jaw"].append(float(t_jaw))
        state_source_times["cmd_arm"].append(float(t_cmd_arm))
        state_source_times["cmd_jaw"].append(float(t_cmd_jaw))

        if is_image_tick:
            for cam in cam_names:
                images[cam].append(tick_images[cam])
                image_source_times[cam].append(tick_source_cams[cam])
            image_timestamps.append(float(t_arm))
            image_indices.append(accepted_steps)

        accepted_steps += 1

    if accepted_steps < 2:
        return None

    capture_ratio = accepted_steps / requested_steps if requested_steps > 0 else 0.0

    return {
        "qpos":               qpos_list,
        "qvel":               qvel_list,
        "action":             action_list,
        "images":             images,
        "timestamps":         timestamps,
        "source_times":       state_source_times,
        "image_timestamps":   image_timestamps,
        "image_indices":      image_indices,
        "image_source_times": image_source_times,
        "arm_effort":         arm_effort_list,
        "spatial_cf":         spatial_cf_list,
        "body_cf":            body_cf_list,
        "measured_cf":        measured_cf_list,
        "joint_force":        joint_force_list,
        "meta": {
            "requested_seconds":     float(requested_seconds or 0.0),
            "requested_fps":         float(fps),
            "requested_image_fps":   float(image_fps),
            "image_interval":        int(image_interval),
            "requested_steps":       int(requested_steps),
            "captured_steps":        int(accepted_steps),
            "dropped_steps":         int(dropped_steps),
            "capture_ratio":         float(capture_ratio),
            "state_tolerance_sec":   float(state_tolerance),
            "image_tolerance_sec":   float(image_tolerance),
            "command_tolerance_sec": float(command_tolerance),
            "wall_start_time":       float(wall_start_time or t_start),
            "wall_end_time":         float(time.time()),
            "joint_dim":             int(len(qpos_list[0])),
            "command_mode":          str(command_mode),
        }
    }


# ============================================================
# HDF5 Dataset Saver
# ============================================================

def save_episode(save_path,
                 data,
                 episode_idx,
                 topic_names,
                 resize=None,
                 image_encoding='bgr8'):
    """Save synchronized multi-modal data into standard ACT HDF5 layout."""
    if h5py is None:
        raise ImportError("h5py is required to save HDF5 files.")

    qpos = np.stack(data["qpos"])
    qvel = np.stack(data["qvel"])
    action = np.stack(data["action"])
    cam_names = list(data["images"].keys())

    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)

    with h5py.File(save_path, 'w') as f:
        f.attrs['sim'] = False
        f.attrs['source'] = 'dvrk'
        f.attrs['episode_idx'] = int(episode_idx)
        f.attrs['fps'] = float(data["meta"]["requested_fps"])
        f.attrs['image_fps'] = float(data["meta"]["requested_image_fps"])
        f.attrs['image_interval'] = int(data["meta"]["image_interval"])
        f.attrs['dt'] = 1.0 / float(data["meta"]["requested_fps"])
        f.attrs['requested_steps'] = int(data["meta"]["requested_steps"])
        f.attrs['captured_steps'] = int(data["meta"]["captured_steps"])
        f.attrs['dropped_steps'] = int(data["meta"]["dropped_steps"])
        f.attrs['capture_ratio'] = float(data["meta"]["capture_ratio"])

        f.attrs['state_tolerance_sec'] = float(data["meta"]["state_tolerance_sec"])
        f.attrs['image_tolerance_sec'] = float(data["meta"]["image_tolerance_sec"])
        f.attrs['command_tolerance_sec'] = float(data["meta"]["command_tolerance_sec"])

        f.attrs['joint_dim'] = int(data["meta"]["joint_dim"])
        f.attrs['wall_start_time'] = float(data["meta"]["wall_start_time"])
        f.attrs['wall_end_time'] = float(data["meta"]["wall_end_time"])
        f.attrs['recorded_at_iso'] = datetime.now().isoformat(timespec="seconds")
        f.attrs['command_mode'] = str(data["meta"]["command_mode"])

        f.attrs['camera_names'] = np.array(cam_names, dtype='S')
        f.attrs['color_order'] = str(image_encoding)
        f.attrs['state_semantics'] = 'PSM_arm[:arm_joints] | PSM_jaw[:jaw_joints]'
        f.attrs['action_semantics'] = 'cmd_arm[:arm_joints] | cmd_jaw[:jaw_joints]'
        f.attrs['qvel_semantics'] = 'PSM_arm_vel[:arm_joints] | PSM_jaw_vel[:jaw_joints]'

        for k, v in topic_names.items():
            f.attrs[k] = v

        if resize is None:
            f.attrs['resize_w'] = 0
            f.attrs['resize_h'] = 0
        else:
            f.attrs['resize_w'] = int(resize[0])
            f.attrs['resize_h'] = int(resize[1])

        obs = f.create_group('observations')
        obs.create_dataset('qpos', data=qpos, dtype='f4')
        obs.create_dataset('qvel', data=qvel, dtype='f4')

        img_group = obs.create_group('images')
        for cam in cam_names:
            img_group.create_dataset(
                cam,
                data=np.stack(data["images"][cam]),
                dtype='u1',
                compression="gzip",
                compression_opts=2
            )

        f.create_dataset('action', data=action, dtype='f4')
        f.create_dataset('timestamp', data=np.asarray(data["timestamps"], dtype=np.float64))
        f.create_dataset('image_timestamp', data=np.asarray(data["image_timestamps"], dtype=np.float64))
        f.create_dataset('image_indices', data=np.asarray(data["image_indices"], dtype=np.int32))

        st = f.create_group("source_timestamps")
        for name, arr in data["source_times"].items():
            st.create_dataset(name, data=np.asarray(arr, dtype=np.float64))

        ist = f.create_group("image_source_timestamps")
        for name, arr in data["image_source_times"].items():
            if arr:
                ist.create_dataset(name, data=np.asarray(arr, dtype=np.float64))

        # Force/Torque & Joint Efforts (ALWAYS COLLECTED)
        f.create_dataset('arm_effort', data=np.stack(data["arm_effort"]), dtype='f4')
        f.create_dataset('spatial_cf', data=np.stack(data["spatial_cf"]), dtype='f4')
        f.create_dataset('body_cf', data=np.stack(data["body_cf"]), dtype='f4')
        f.create_dataset('measured_cf', data=np.stack(data["measured_cf"]), dtype='f4')
        f.create_dataset('joint_force', data=np.stack(data["joint_force"]), dtype='f4')

    print(f"[HDF5] Saved episode: {save_path}")


# ============================================================
# Interactive Terminal Runner (CLI / Headless Mode)
# ============================================================

def run_cli_recording(args, node):
    """Fallback interactive CLI loop when running headless or with --no-gui."""
    os.makedirs(args.output_dir, exist_ok=True)
    if args.record_bag:
        os.makedirs(args.bag_dir, exist_ok=True)
    if args.force_sensor:
        os.makedirs(args.force_bag_dir, exist_ok=True)

    episode_idx = get_next_episode_idx(args.output_dir)
    print(f"\n[CLI Mode] Starting data collection from episode index {episode_idx}")
    print(f"PSM: {args.psm} | State FPS: {args.fps} | Camera FPS: {args.image_fps} | Duration: {args.seconds_per_episode}s")

    resize = (args.resize_w, args.resize_h) if (args.resize_w > 0 and args.resize_h > 0) else None

    try:
        while True:
            wait_for_key('q')
            print(f"\nRecording episode {episode_idx} ({args.seconds_per_episode}s) ...")
            print("\a")

            bag_path = None
            force_bag_path = None
            bag_proc = None
            force_bag_proc = None

            topics = node.topic_names
            if args.record_bag:
                bag_path = os.path.join(args.bag_dir, f'episode_{episode_idx:03d}_bag')
                bag_proc = start_rosbag_recording(bag_path, list(topics.values()))

            if args.force_sensor:
                force_bag_path = os.path.join(args.force_bag_dir, f'episode_{episode_idx:03d}_force')
                force_bag_topics = [
                    topics["measured_cf_topic"],
                    topics["gravity_topic"],
                    topics["psm_arm_topic"],
                    topics["spatial_cf_topic"],
                    topics["body_cf_topic"],
                ]
                force_bag_proc = start_rosbag_recording(force_bag_path, force_bag_topics)

            node.start_recording()

            time.sleep(args.seconds_per_episode)

            buffers, wall_start = node.stop_recording()

            if bag_proc: stop_rosbag_recording(bag_proc)
            if force_bag_proc: stop_rosbag_recording(force_bag_proc)

            print("Synchronizing trial timesteps...")
            data = synchronize_trial_data(
                trial_buffers=buffers,
                fps=args.fps,
                image_fps=args.image_fps,
                state_tolerance=args.state_tolerance,
                image_tolerance=args.image_tolerance,
                command_tolerance=args.command_tolerance,
                requested_seconds=args.seconds_per_episode,
                resize=resize,
                command_mode=args.command_mode,
                wrist_cameras=args.wrist_cameras,
                wall_start_time=wall_start
            )

            if data is None or len(data.get("qpos", [])) < 2:
                print("[ERROR] Trial failed or too many dropped frames. Discarding automatically.")
                if bag_path and os.path.exists(bag_path): shutil.rmtree(bag_path, ignore_errors=True)
                if force_bag_path and os.path.exists(force_bag_path): shutil.rmtree(force_bag_path, ignore_errors=True)
                continue

            meta = data["meta"]
            captured = meta["captured_steps"]
            requested = meta["requested_steps"]
            dropped = meta["dropped_steps"]
            ratio = meta["capture_ratio"] * 100.0

            print(f"[Trial Summary] Captured {captured}/{requested} steps ({ratio:.1f}%), {dropped} dropped.")

            # Operator choice: Save or Discard
            choice = input(f"Save episode_{episode_idx:03d}? [Y/n]: ").strip().lower()
            if choice in ('', 'y', 'yes'):
                save_path = os.path.join(args.output_dir, f'episode_{episode_idx:03d}.hdf5')
                save_episode(
                    save_path=save_path,
                    data=data,
                    episode_idx=episode_idx,
                    topic_names=node.topic_names,
                    resize=resize,
                    image_encoding=args.image_encoding
                )
                print(f"Episode {episode_idx} saved.")
                episode_idx += 1
            else:
                print(f"Episode {episode_idx} discarded.")
                if bag_path and os.path.exists(bag_path): shutil.rmtree(bag_path, ignore_errors=True)
                if force_bag_path and os.path.exists(force_bag_path): shutil.rmtree(force_bag_path, ignore_errors=True)

    except KeyboardInterrupt:
        print("\nStopped by user.")


# ============================================================
# Argument Parsing & Entry Point
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Record dVRK teleoperation episodes with high-speed async ROS 2 intake and optional PyQt GUI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument('--output_dir', default='./dvrk_normal_dataset',
                        help='Directory where HDF5 episodes are saved')
    parser.add_argument('--seconds_per_episode', type=float, default=15.0,
                        help='Duration of each episode in seconds')
    parser.add_argument('--fps', type=float, default=100.0,
                        help='Target state/action/force recording frequency (Hz)')
    parser.add_argument('--image_fps', type=float, default=20.0,
                        help='Target camera recording frequency (Hz). Must be <= --fps')

    parser.add_argument('--state_tolerance', type=float, default=0.03,
                        help='Nearest-sample tolerance for state/force topics (seconds)')
    parser.add_argument('--image_tolerance', type=float, default=0.08,
                        help='Nearest-frame tolerance for cameras (seconds)')
    parser.add_argument('--command_tolerance', type=float, default=0.05,
                        help='Tolerance for operator joint setpoints (seconds)')
    parser.add_argument('--warmup_sec', type=float, default=0.0,
                        help='Warmup delay before recording begins')

    # Robot arm selection
    parser.add_argument('--psm', type=str, default='PSM2', choices=['PSM1', 'PSM2'],
                        help='dVRK Patient Side Manipulator arm identifier')
    parser.add_argument('--arm_joints', type=int, default=6,
                        help='Number of arm joints (excluding jaw)')
    parser.add_argument('--jaw_joints', type=int, default=1,
                        help='Number of jaw joints')

    # Camera topics
    parser.add_argument('--left_topic', default='/console1/left/image_raw')
    parser.add_argument('--right_topic', default='/console1/right/image_raw')
    parser.add_argument('--wrist1_topic', default='/wrist/cam1/image_raw')
    parser.add_argument('--wrist2_topic', default='/wrist/cam2/image_raw')
    parser.add_argument('--wrist_cameras', nargs='*', default=['cam1', 'cam2'], choices=['cam1', 'cam2'],
                        help='Which wrist cameras to include')

    # Image formatting
    parser.add_argument('--resize_w', type=int, default=320)
    parser.add_argument('--resize_h', type=int, default=240)
    parser.add_argument('--image_encoding', choices=['bgr8', 'rgb8'], default='bgr8')

    parser.add_argument('--command_mode', choices=['latest_before_or_nearest', 'nearest'],
                        default='latest_before_or_nearest')

    # Optional rosbag backups
    parser.add_argument('--record_bag', action='store_true',
                        help='Also record a raw rosbag backup per trial')
    parser.add_argument('--bag_dir', default='./dvrk_bags')
    parser.add_argument('--force_sensor', action='store_true',
                        help='Also record a separate force-sensor rosbag per trial')
    parser.add_argument('--force_bag_dir', default='./dvrk_force_bags')

    # Interface mode
    parser.add_argument('--gui', action='store_true',
                        help='Launch the PyQt Graphical User Interface (record_gui.py)')
    parser.add_argument('--no-gui', '--headless', dest='headless', action='store_true',
                        help='Run in interactive terminal mode instead of launching GUI')

    return parser.parse_args()


def main():
    args = parse_args()

    if args.image_fps > args.fps:
        print(f"[ERROR] --image_fps ({args.image_fps}) must be <= --fps ({args.fps})")
        sys.exit(1)

    # Launch GUI if explicitly requested via --gui or by default when not in headless mode
    if args.gui or (not args.headless):
        try:
            from record_gui import launch_gui, QT_AVAILABLE
            if QT_AVAILABLE:
                launch_gui(args)
                return
        except ImportError:
            pass

    # Fallback to CLI
    if not ROS_AVAILABLE:
        print("[ERROR] rclpy is required for CLI recording.")
        sys.exit(1)

    rclpy.init()
    node = DVRKRecorder(
        psm=args.psm,
        wrist_cameras=args.wrist_cameras,
        left_topic=args.left_topic,
        right_topic=args.right_topic,
        wrist1_topic=args.wrist1_topic,
        wrist2_topic=args.wrist2_topic,
        arm_joints=args.arm_joints,
        jaw_joints=args.jaw_joints,
        image_encoding=args.image_encoding,
    )

    executor = SingleThreadedExecutor()
    executor.add_node(node)
    ros_thread = threading.Thread(target=executor.spin, daemon=True)
    ros_thread.start()

    run_cli_recording(args, node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()
