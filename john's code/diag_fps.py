#!/usr/bin/env python3
"""Diagnose 16Hz bottleneck: differentiate "camera/driver limitation" vs "ROS node overhead".
Usage: python3 diag_fps.py /dev/v4l/by-id/xxx
"""
import statistics
import sys
import time

import cv2

N_FRAMES = 120
N_WARMUP = 15


def fourcc_str(value):
    v = int(value)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4))


def force_manual_exposure(cap):
    """OpenCV V4L2 backend has two value conventions; try both and return the one in effect."""
    for probe in (1, 0.25):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, probe)
        if abs(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) - probe) < 1e-3:
            cap.set(cv2.CAP_PROP_EXPOSURE, 50)
            return probe
    return None


def measure(device, label, setup):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[{label}] Failed to open\n")
        return
    note = setup(cap) or ""

    for _ in range(N_WARMUP):
        cap.read()

    dts = []
    n_fail = 0
    t_prev = time.perf_counter()
    for _ in range(N_FRAMES):
        ok, _ = cap.read()
        t_now = time.perf_counter()
        if not ok:
            n_fail += 1
        dts.append((t_now - t_prev) * 1000.0)
        t_prev = t_now

    got = (f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
           f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    fcc = fourcc_str(cap.get(cv2.CAP_PROP_FOURCC))
    cap.release()

    ordered = sorted(dts)
    print(f"[{label}] {note}")
    print(f"  Effective format    : {fcc} {got}")
    print(f"  Average FPS         : {1000.0 / statistics.mean(dts):6.1f} Hz")
    print(f"  Median frame interval: {statistics.median(dts):6.1f} ms  "
          f"(min {ordered[0]:.1f} / max {ordered[-1]:.1f})")
    print(f"  Dropped frames      : {n_fail}/{N_FRAMES}\n")


def setup_baseline(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return "MJPG 640x480 Auto Exposure"


def setup_manual_exposure(cap):
    setup_baseline(cap)
    used = force_manual_exposure(cap)
    if used is None:
        return "MJPG 640x480 Manual Exposure (camera rejected control, invalid)"
    return f"MJPG 640x480 Manual Exposure (convention {used}, exposure=50)"


def setup_small(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return "MJPG 320x240 Auto Exposure"


def setup_yuyv(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return "YUYV 640x480 Auto Exposure"


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
    print(f"Device: {dev}\n")
    measure(dev, "A Baseline", setup_baseline)
    measure(dev, "B Manual Exposure", setup_manual_exposure)
    measure(dev, "C Lower Resolution", setup_small)
    measure(dev, "D Switch to YUYV", setup_yuyv)