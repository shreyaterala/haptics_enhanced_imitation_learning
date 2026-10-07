#!/usr/bin/env python3
"""纯标准库 V4L2 探测/设置工具（无需 sudo、无需 v4l-utils、不依赖 OpenCV）。

用法:
  python3 v4l2_probe.py /dev/video0                    # 列出格式、分辨率、帧率、控制项
  python3 v4l2_probe.py /dev/video0 --fix-framerate    # exposure_auto_priority=0，禁止驱动为曝光降帧
  python3 v4l2_probe.py /dev/video0 --manual-exposure 300
  python3 v4l2_probe.py /dev/video0 --auto-exposure    # 恢复自动曝光
  python3 v4l2_probe.py /dev/video0 --set 0x009A0903=0 # 手动设任意控制项
"""
import argparse
import errno
import fcntl
import os
import struct
import sys

BUF_TYPE_VIDEO_CAPTURE = 1

VIDIOC_ENUM_FMT = 0xC0405602
VIDIOC_ENUM_FRAMESIZES = 0xC02C564A
VIDIOC_ENUM_FRAMEINTERVALS = 0xC034564B
VIDIOC_QUERYCTRL = 0xC0445624
VIDIOC_G_CTRL = 0xC008561B
VIDIOC_S_CTRL = 0xC008561C

FMT_FLAG_COMPRESSED = 0x0001

FRMSIZE_DISCRETE, FRMSIZE_CONTINUOUS, FRMSIZE_STEPWISE = 1, 2, 3
FRMIVAL_DISCRETE, FRMIVAL_CONTINUOUS, FRMIVAL_STEPWISE = 1, 2, 3

CTRL_TYPE_NAMES = {
    1: "integer", 2: "boolean", 3: "menu", 4: "button",
    5: "integer64", 6: "ctrl_class", 7: "string", 8: "bitmask",
    9: "integer_menu",
}

# 与帧率相关的控制项优先显示
CONTROLS = [
    ("brightness", 0x00980900),
    ("contrast", 0x00980901),
    ("gain", 0x00980913),
    ("power_line_frequency", 0x00980918),
    ("exposure_auto", 0x009A0901),
    ("exposure_absolute", 0x009A0902),
    ("exposure_auto_priority", 0x009A0903),
]

EXPOSURE_AUTO_MODE = 0
EXPOSURE_MANUAL_MODE = 1


def fourcc_to_str(value):
    raw = struct.pack("<I", value)
    return "".join(chr(b) if 32 <= b < 127 else "?" for b in raw)


def enum_formats(fd):
    formats = []
    for index in range(64):
        buf = bytearray(struct.pack("<3I32s5I", index, BUF_TYPE_VIDEO_CAPTURE,
                                    0, b"", 0, 0, 0, 0, 0))
        try:
            fcntl.ioctl(fd, VIDIOC_ENUM_FMT, buf)
        except OSError:
            break
        _, _, flags, desc, pixfmt = struct.unpack("<3I32s5I", buf)[:5]
        formats.append({
            "pixelformat": pixfmt,
            "fourcc": fourcc_to_str(pixfmt),
            "description": desc.split(b"\x00")[0].decode("utf-8", "replace"),
            "compressed": bool(flags & FMT_FLAG_COMPRESSED),
        })
    return formats


def enum_frame_sizes(fd, pixfmt):
    sizes = []
    for index in range(64):
        buf = bytearray(struct.pack("<11I", index, pixfmt, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0))
        try:
            fcntl.ioctl(fd, VIDIOC_ENUM_FRAMESIZES, buf)
        except OSError:
            break
        vals = struct.unpack("<11I", buf)
        stype = vals[2]
        if stype == FRMSIZE_DISCRETE:
            sizes.append((vals[3], vals[4], "discrete"))
        else:
            # stepwise/continuous: min_w,max_w,step_w,min_h,max_h,step_h
            sizes.append((vals[4], vals[7], "stepwise"))
            break
    return sizes


def enum_frame_intervals(fd, pixfmt, width, height):
    out = []
    for index in range(64):
        buf = bytearray(struct.pack("<13I", index, pixfmt, width, height, 0,
                                    0, 0, 0, 0, 0, 0, 0, 0))
        try:
            fcntl.ioctl(fd, VIDIOC_ENUM_FRAMEINTERVALS, buf)
        except OSError:
            break
        vals = struct.unpack("<13I", buf)
        itype = vals[4]
        if itype == FRMIVAL_DISCRETE:
            num, den = vals[5], vals[6]
            if num:
                out.append(den / num)
        else:
            # stepwise: min(num,den), max(num,den), step(num,den)
            min_num, min_den, max_num, max_den = vals[5], vals[6], vals[7], vals[8]
            if min_num and max_num:
                out.append(min_den / min_num)   # 间隔最小 = 帧率最高
                out.append(max_den / max_num)
            break
    return out


def query_control(fd, cid):
    buf = bytearray(struct.pack("<2I32s4iI2I", cid, 0, b"", 0, 0, 0, 0, 0, 0, 0))
    try:
        fcntl.ioctl(fd, VIDIOC_QUERYCTRL, buf)
    except OSError:
        return None
    cid_out, ctype, name, cmin, cmax, cstep, cdef, flags = \
        struct.unpack("<2I32s4iI2I", buf)[:8]
    return {
        "id": cid_out,
        "type": ctype,
        "name": name.split(b"\x00")[0].decode("utf-8", "replace"),
        "min": cmin, "max": cmax, "step": cstep, "default": cdef,
        "disabled": bool(flags & 0x0001),
        "read_only": bool(flags & 0x0002),
    }


def get_control(fd, cid):
    buf = bytearray(struct.pack("<Ii", cid, 0))
    try:
        fcntl.ioctl(fd, VIDIOC_G_CTRL, buf)
    except OSError:
        return None
    return struct.unpack("<Ii", buf)[1]


def set_control(fd, cid, value):
    buf = bytearray(struct.pack("<Ii", cid, int(value)))
    try:
        fcntl.ioctl(fd, VIDIOC_S_CTRL, buf)
    except OSError as exc:
        reason = "设备不支持该控制项" if exc.errno == errno.EINVAL else exc.strerror
        print(f"  设置 0x{cid:08X} = {value} 失败: {reason}")
        return False
    actual = get_control(fd, cid)
    print(f"  设置 0x{cid:08X} = {value} 成功（回读 {actual}）")
    return True


def report(fd, device):
    print(f"设备: {device}\n")

    formats = enum_formats(fd)
    if not formats:
        print("没有枚举到任何采集格式 —— 这个节点可能是 metadata 节点，"
              "换 by-id 里 index 更小的那个。")
        return

    print("=== 支持的格式 / 分辨率 / 帧率 ===")
    for fmt in formats:
        tag = "压缩" if fmt["compressed"] else "未压缩"
        print(f"\n[{fmt['fourcc']}] {fmt['description']} ({tag})")
        sizes = enum_frame_sizes(fd, fmt["pixelformat"])
        if not sizes:
            print("  （驱动未枚举分辨率）")
            continue
        for width, height, kind in sizes[:24]:
            fps_list = enum_frame_intervals(fd, fmt["pixelformat"], width, height)
            if fps_list:
                shown = ", ".join(f"{f:g}" for f in sorted(fps_list, reverse=True))
            else:
                shown = "未枚举"
            suffix = "  <-- 步进范围" if kind == "stepwise" else ""
            print(f"  {width:>5}x{height:<5}  fps: {shown}{suffix}")
        if len(sizes) > 24:
            print(f"  …还有 {len(sizes) - 24} 组分辨率未显示")

    print("\n=== 与帧率/亮度相关的控制项 ===")
    for label, cid in CONTROLS:
        info = query_control(fd, cid)
        if info is None:
            print(f"  {label:<24} 不支持")
            continue
        current = get_control(fd, cid)
        ctype = CTRL_TYPE_NAMES.get(info["type"], str(info["type"]))
        flags = []
        if info["disabled"]:
            flags.append("当前不可用")
        if info["read_only"]:
            flags.append("只读")
        extra = ("  [" + ", ".join(flags) + "]") if flags else ""
        print(f"  {label:<24} 当前={current}  范围={info['min']}..{info['max']}"
              f"  默认={info['default']}  类型={ctype}"
              f"  id=0x{cid:08X}{extra}")

    print("\n关注两点：MJPG 那一块在 640x480 有没有 30；"
          "以及 exposure_auto_priority 当前是不是 1。")


def main():
    ap = argparse.ArgumentParser(description="纯标准库 V4L2 探测/设置工具")
    ap.add_argument("device", nargs="?", default="/dev/video0")
    ap.add_argument("--fix-framerate", action="store_true",
                    help="exposure_auto_priority=0，禁止驱动为了曝光而降帧")
    ap.add_argument("--manual-exposure", type=int, metavar="VALUE",
                    help="锁定手动曝光并设置 exposure_absolute")
    ap.add_argument("--auto-exposure", action="store_true",
                    help="恢复自动曝光")
    ap.add_argument("--set", action="append", default=[], metavar="ID=VAL",
                    help="设置任意控制项，如 --set 0x009A0903=0")
    args = ap.parse_args()

    if not os.path.exists(args.device):
        print(f"设备不存在: {args.device}")
        return 1

    try:
        fd = os.open(args.device, os.O_RDWR)
    except OSError as exc:
        print(f"打开 {args.device} 失败: {exc.strerror}"
              f"（检查是否在 video 组、设备是否被占用）")
        return 1

    try:
        changed = False

        if args.auto_exposure:
            print("恢复自动曝光:")
            set_control(fd, 0x009A0901, EXPOSURE_AUTO_MODE)
            changed = True

        if args.manual_exposure is not None:
            print("锁定手动曝光:")
            set_control(fd, 0x009A0901, EXPOSURE_MANUAL_MODE)
            set_control(fd, 0x009A0902, args.manual_exposure)
            changed = True

        if args.fix_framerate:
            print("禁止驱动为曝光降帧:")
            set_control(fd, 0x009A0903, 0)
            changed = True

        for item in args.set:
            if "=" not in item:
                print(f"忽略无法解析的 --set 参数: {item}")
                continue
            key, val = item.split("=", 1)
            print(f"设置 {key}:")
            set_control(fd, int(key, 0), int(val, 0))
            changed = True

        if changed:
            print()
        report(fd, args.device)
    finally:
        os.close(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())