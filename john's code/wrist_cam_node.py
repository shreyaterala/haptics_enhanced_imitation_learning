#!/usr/bin/env python3
"""ROS2 publisher node for USB camera without requiring sudo.
Publishes: <ns>/image_raw, <ns>/image_raw/compressed, <ns>/camera_info
Usage: python3 wrist_cam_node.py --ros-args -p video_device:=/dev/video0
"""
import os
import sys

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from sensor_msgs.msg import CameraInfo, CompressedImage, Image


class WristCam(Node):
    def __init__(self):
        super().__init__("wrist_cam")

        self.declare_parameter("video_device", "/dev/video0")
        self.declare_parameter("fourcc", "MJPG")
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("framerate", 30.0)
        self.declare_parameter("frame_id", "wrist_camera_optical_frame")
        self.declare_parameter("camera_info_file", "")
        self.declare_parameter("publish_compressed", True)
        self.declare_parameter("jpeg_quality", 85)

        p = self.get_parameter
        self.device = p("video_device").value
        self.frame_id = p("frame_id").value
        self.fps = float(p("framerate").value)
        self.pub_comp = bool(p("publish_compressed").value)
        self.jpeg_quality = int(p("jpeg_quality").value)

        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Failed to open {self.device}: check if path exists and current user is in video group")

        fourcc = str(p("fourcc").value)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(p("image_width").value))
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(p("image_height").value))
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # Reduce queuing latency

        got_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        got_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.get_logger().info(
            f"{self.device}: {fourcc} {got_w}x{got_h} "
            f"@ {self.cap.get(cv2.CAP_PROP_FPS):.1f} fps")

        self.info_msg = self._load_camera_info(
            p("camera_info_file").value, got_w, got_h)

        self.pub_img = self.create_publisher(Image, "image_raw", 1)
        self.pub_jpg = self.create_publisher(
            CompressedImage, "image_raw/compressed", 1)
        self.pub_info = self.create_publisher(CameraInfo, "camera_info", 1)

        self.n_dropped = 0
        self.timer = self.create_timer(1.0 / self.fps, self.tick)

    def _load_camera_info(self, path, width, height):
        msg = CameraInfo()
        msg.width, msg.height = width, height
        msg.distortion_model = "plumb_bob"
        if not path:
            self.get_logger().warn("camera_info_file not provided, intrinsics empty (normal before calibration)")
            return msg
        path = os.path.expanduser(path)
        if not os.path.isfile(path):
            self.get_logger().warn(f"Calibration file {path} not found, intrinsics empty")
            return msg
        with open(path, "r") as f:
            c = yaml.safe_load(f)
        msg.width = c["image_width"]
        msg.height = c["image_height"]
        msg.distortion_model = c.get("distortion_model", "plumb_bob")
        msg.d = list(c["distortion_coefficients"]["data"])
        msg.k = list(c["camera_matrix"]["data"])
        msg.r = list(c["rectification_matrix"]["data"])
        msg.p = list(c["projection_matrix"]["data"])
        self.get_logger().info(f"Loaded intrinsics: {path}")
        return msg

    def tick(self):
        ok, frame = self.cap.read()
        if not ok:
            self.n_dropped += 1
            if self.n_dropped % 30 == 1:
                self.get_logger().warn(f"Failed to grab frame x{self.n_dropped}")
            return

        stamp = self.get_clock().now().to_msg()

        img = Image()
        img.header.stamp = stamp
        img.header.frame_id = self.frame_id
        img.height, img.width = frame.shape[0], frame.shape[1]
        img.encoding = "bgr8"
        img.is_bigendian = 0
        img.step = img.width * 3
        img.data = np.ascontiguousarray(frame).tobytes()
        self.pub_img.publish(img)

        if self.pub_comp:
            ok_enc, buf = cv2.imencode(
                ".jpg", frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality])
            if ok_enc:
                jpg = CompressedImage()
                jpg.header = img.header
                jpg.format = "jpeg"
                jpg.data = buf.tobytes()
                self.pub_jpg.publish(jpg)

        self.info_msg.header = img.header
        self.pub_info.publish(self.info_msg)

    def destroy_node(self):
        if self.cap is not None:
            self.cap.release()
        super().destroy_node()


def main():
    rclpy.init(args=sys.argv)
    node = None
    try:
        node = WristCam()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()