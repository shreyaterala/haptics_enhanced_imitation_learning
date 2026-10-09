#!/usr/bin/env python3
"""Pure standard-library V4L2 probe/configuration tool (no sudo, no v4l-utils, no OpenCV dependency).

Usage:
  python3 v4l2_probe.py /dev/video0                    # List formats, resolutions, frame rates, controls
  python3 v4l2_probe.py /dev/video0 --fix-framerate    # exposure_auto_priority=0, prevent driver from dropping FPS for exposure
  python3 v4l2_probe.py /dev/video0 --manual-exposure 300
  python3 v4l2_probe.py /dev/video0 --auto-exposure    # Restore auto exposure
  python3 v4l2_probe.py /dev/video0 --set 0x009A0903=0 # Manually set arbitrary control
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

# Prioritize controls related to frame rate
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
                out.append(min_den / min_num)   # Smallest interval = highest frame rate
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
        reason = "Device does not support this control" if exc.errno == errno.EINVAL else exc.strerror
        print(f"  Setting 0x{cid:08X} = {value} failed: {reason}")
        return False
    actual = get_control(fd, cid)
    print(f"  Setting 0x{cid:08X} = {value} succeeded (read back {actual})")
    return True


def report(fd, device):
    print(f"Device: {device}\n")

    formats = enum_formats(fd)
    if not formats:
        print("No capture formats enumerated — this node may be a metadata node, "
              "try the one with smaller index in by-id.")
        return

    print("=== Supported Formats / Resolutions / Frame Rates ===")
    for fmt in formats:
        tag = "compressed" if fmt["compressed"] else "uncompressed"
        print(f"\n[{fmt['fourcc']}] {fmt['description']} ({tag})")
        sizes = enum_frame_sizes(fd, fmt["pixelformat"])
        if not sizes:
            print("  (Driver did not enumerate resolutions)")
            continue
        for width, height, kind in sizes[:24]:
            fps_list = enum_frame_intervals(fd, fmt["pixelformat"], width, height)
            if fps_list:
                shown = ", ".join(f"{f:g}" for f in sorted(fps_list, reverse=True))
            else:
                shown = "Not enumerated"
            suffix = "  <-- Stepwise range" if kind == "stepwise" else ""
            print(f"  {width:>5}x{height:<5}  fps: {shown}{suffix}")
        if len(sizes) > 24:
            print(f"  ...and {len(sizes) - 24} more resolution sets not shown")

    print("\n=== Frame Rate / Exposure Related Controls ===")
    for label, cid in CONTROLS:
        info = query_control(fd, cid)
        if info is None:
            print(f"  {label:<24} Not supported")
            continue
        current = get_control(fd, cid)
        ctype = CTRL_TYPE_NAMES.get(info["type"], str(info["type"]))
        flags = []
        if info["disabled"]:
            flags.append("Currently unavailable")
        if info["read_only"]:
            flags.append("Read-only")
        extra = ("  [" + ", ".join(flags) + "]") if flags else ""
        print(f"  {label:<24} Current={current}  Range={info['min']}..{info['max']}"
              f"  Default={info['default']}  Type={ctype}"
              f"  id=0x{cid:08X}{extra}")

    print("\nKey checkpoints: check if MJPG at 640x480 supports 30 fps; "
          "and whether exposure_auto_priority is currently 1.")


def main():
    ap = argparse.ArgumentParser(description="Pure standard-library V4L2 probe/configuration tool")
    ap.add_argument("device", nargs="?", default="/dev/video0")
    ap.add_argument("--fix-framerate", action="store_true",
                    help="exposure_auto_priority=0, prevent driver from dropping FPS for exposure")
    ap.add_argument("--manual-exposure", type=int, metavar="VALUE",
                    help="Lock manual exposure and set exposure_absolute")
    ap.add_argument("--auto-exposure", action="store_true",
                    help="Restore auto exposure")
    ap.add_argument("--set", action="append", default=[], metavar="ID=VAL",
                    help="Set arbitrary control, e.g. --set 0x009A0903=0")
    args = ap.parse_args()

    if not os.path.exists(args.device):
        print(f"Device does not exist: {args.device}")
        return 1

    try:
        fd = os.open(args.device, os.O_RDWR)
    except OSError as exc:
        print(f"Failed to open {args.device}: {exc.strerror}"
              f" (check video group permissions or if device is in use)")
        return 1

    try:
        changed = False

        if args.auto_exposure:
            print("Restoring auto exposure:")
            set_control(fd, 0x009A0901, EXPOSURE_AUTO_MODE)
            changed = True

        if args.manual_exposure is not None:
            print("Locking manual exposure:")
            set_control(fd, 0x009A0901, EXPOSURE_MANUAL_MODE)
            set_control(fd, 0x009A0902, args.manual_exposure)
            changed = True

        if args.fix_framerate:
            print("Preventing driver from dropping FPS for exposure:")
            set_control(fd, 0x009A0903, 0)
            changed = True

        for item in args.set:
            if "=" not in item:
                print(f"Ignoring unparseable --set parameter: {item}")
                continue
            key, val = item.split("=", 1)
            print(f"Setting {key}:")
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