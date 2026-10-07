#!/usr/bin/env python3
"""定位 16Hz 瓶颈：区分"相机/驱动本身出不来" vs "ROS 节点拖慢"。
用法: python3 diag_fps.py /dev/v4l/by-id/xxx
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
    """OpenCV 的 V4L2 后端有两套取值约定，都试一遍，返回实际生效的那个。"""
    for probe in (1, 0.25):
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, probe)
        if abs(cap.get(cv2.CAP_PROP_AUTO_EXPOSURE) - probe) < 1e-3:
            cap.set(cv2.CAP_PROP_EXPOSURE, 50)
            return probe
    return None


def measure(device, label, setup):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        print(f"[{label}] 打开失败\n")
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
    print(f"  实际生效格式 : {fcc} {got}")
    print(f"  平均帧率     : {1000.0 / statistics.mean(dts):6.1f} Hz")
    print(f"  帧间隔中位   : {statistics.median(dts):6.1f} ms  "
          f"(最小 {ordered[0]:.1f} / 最大 {ordered[-1]:.1f})")
    print(f"  取帧失败     : {n_fail}/{N_FRAMES}\n")


def setup_baseline(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return "MJPG 640x480 自动曝光"


def setup_manual_exposure(cap):
    setup_baseline(cap)
    used = force_manual_exposure(cap)
    if used is None:
        return "MJPG 640x480 手动曝光（相机不接受该控制，本项无效）"
    return f"MJPG 640x480 手动曝光（约定值 {used}，曝光=50）"


def setup_small(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return "MJPG 320x240 自动曝光"


def setup_yuyv(cap):
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return "YUYV 640x480 自动曝光"


if __name__ == "__main__":
    dev = sys.argv[1] if len(sys.argv) > 1 else "/dev/video0"
    print(f"设备: {dev}\n")
    measure(dev, "A 基线", setup_baseline)
    measure(dev, "B 手动曝光", setup_manual_exposure)
    measure(dev, "C 更低分辨率", setup_small)
    measure(dev, "D 换 YUYV", setup_yuyv)