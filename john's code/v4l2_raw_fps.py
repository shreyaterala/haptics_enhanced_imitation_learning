#!/usr/bin/env python3
"""Pure standard-library V4L2 raw stream benchmark (bypassing OpenCV) and control enumeration.

Usage:
  python3 v4l2_raw_fps.py /dev/video0
  python3 v4l2_raw_fps.py /dev/video0 --fps 30 --size 640x480
  python3 v4l2_raw_fps.py /dev/video0 --fps 30 --frames 200
  python3 v4l2_raw_fps.py /dev/video0 --controls-only
"""
import argparse
import errno
import fcntl
import mmap
import os
import select
import statistics
import struct
import sys
import time

BUF_TYPE_VIDEO_CAPTURE = 1
MEMORY_MMAP = 1
FIELD_ANY = 0

VIDIOC_S_FMT = 0xC0D05605
VIDIOC_G_FMT = 0xC0D05604
VIDIOC_REQBUFS = 0xC0145608
VIDIOC_QUERYBUF = 0xC0585609
VIDIOC_QBUF = 0xC058560F
VIDIOC_DQBUF = 0xC0585611
VIDIOC_STREAMON = 0x40045612
VIDIOC_STREAMOFF = 0x40045613
VIDIOC_G_PARM = 0xC0CC5615
VIDIOC_S_PARM = 0xC0CC5616
VIDIOC_QUERYCTRL = 0xC0445624
VIDIOC_G_CTRL = 0xC008561B
VIDIOC_S_CTRL = 0xC008561C

CTRL_FLAG_NEXT_CTRL = 0x80000000
CTRL_FLAG_DISABLED = 0x0001
CTRL_FLAG_READ_ONLY = 0x0002
CTRL_FLAG_INACTIVE = 0x0010
CTRL_FLAG_VOLATILE = 0x0080

CAP_TIMEPERFRAME = 0x1000

CTRL_TYPE_NAMES = {
    1: "int", 2: "bool", 3: "menu", 4: "button", 5: "int64",
    6: "class", 7: "string", 8: "bitmask", 9: "intmenu",
}

# ---- Struct Layout -------------------------------------------------------
# v4l2_format:      type(4) + pad(4) + v4l2_pix_format(48) + pad = 208
FMT_STRUCT = "<I4x12I152x"
# v4l2_streamparm:  type(4) + raw_data(200) = 204
PARM_STRUCT = "<8I172x"
# v4l2_requestbuffers: 5 fields = 20
REQBUFS_STRUCT = "<4IB3x"
# v4l2_buffer = 88, timecode has 4 u8s + userbits[4]
BUF_STRUCT = "<5I4x2q2I4B4s2IQ2Ii4x"
# v4l2_queryctrl = 68
QUERYCTRL_STRUCT = "<2I32s4iI2I"

BUF_FIELDS = (
    "index", "type", "bytesused", "flags", "field",
    "tv_sec", "tv_usec",
    "tc_type", "tc_flags", "tc_frames", "tc_seconds", "tc_minutes",
    "tc_hours", "tc_userbits",
    "sequence", "memory", "m", "length", "reserved2", "request_fd",
)


def ioctl_size(code):
    """Extract expected struct byte size from ioctl encoding."""
    return (code >> 16) & 0x3FFF


def _self_check():
    expected = [
        ("FMT_STRUCT", FMT_STRUCT, ioctl_size(VIDIOC_S_FMT)),
        ("PARM_STRUCT", PARM_STRUCT, ioctl_size(VIDIOC_G_PARM)),
        ("REQBUFS_STRUCT", REQBUFS_STRUCT, ioctl_size(VIDIOC_REQBUFS)),
        ("BUF_STRUCT", BUF_STRUCT, ioctl_size(VIDIOC_QUERYBUF)),
        ("QUERYCTRL_STRUCT", QUERYCTRL_STRUCT, ioctl_size(VIDIOC_QUERYCTRL)),
    ]
    for name, fmt, want in expected:
        got = struct.calcsize(fmt)
        if got != want:
            raise SystemExit(
                f"Internal error: {name} size {got} bytes, kernel expects {want} bytes."
                f"Do not run, or ioctl will write out of memory bounds.")
    n_fields = len(struct.unpack(BUF_STRUCT, b"\x00" * struct.calcsize(BUF_STRUCT)))
    if n_fields != len(BUF_FIELDS):
        raise SystemExit(
            f"Internal error: BUF_STRUCT unpacked {n_fields} fields, "
            f"BUF_FIELDS listed {len(BUF_FIELDS)}.")


_self_check()


def fourcc_from_str(text):
    return struct.unpack("<I", text.ljust(4)[:4].encode())[0]


def fourcc_to_str(value):
    raw = struct.pack("<I", int(value))
    return "".join(chr(b) if 32 <= b < 127 else "?" for b in raw)


def err_name(exc):
    return errno.errorcode.get(exc.errno, str(exc.errno))


def pack_buffer(index=0):
    values = {name: 0 for name in BUF_FIELDS}
    values["index"] = index
    values["type"] = BUF_TYPE_VIDEO_CAPTURE
    values["memory"] = MEMORY_MMAP
    values["tc_userbits"] = b""
    return bytearray(struct.pack(BUF_STRUCT,
                                 *(values[name] for name in BUF_FIELDS)))


def unpack_buffer(raw):
    return dict(zip(BUF_FIELDS, struct.unpack(BUF_STRUCT, raw)))


# ---------------------------------------------------------------- Controls

def query_ctrl(fd, cid):
    buf = bytearray(struct.pack(QUERYCTRL_STRUCT, cid, 0, b"",
                                0, 0, 0, 0, 0, 0, 0))
    try:
        fcntl.ioctl(fd, VIDIOC_QUERYCTRL, buf)
    except OSError:
        return None
    out = struct.unpack(QUERYCTRL_STRUCT, buf)
    return {
        "id": out[0], "type": out[1],
        "name": out[2].split(b"\x00")[0].decode("utf-8", "replace"),
        "min": out[3], "max": out[4], "step": out[5],
        "default": out[6], "flags": out[7],
    }


def get_ctrl(fd, cid):
    buf = bytearray(struct.pack("<Ii", cid, 0))
    try:
        fcntl.ioctl(fd, VIDIOC_G_CTRL, buf)
    except OSError as exc:
        return None, err_name(exc)
    return struct.unpack("<Ii", buf)[1], None


def set_ctrl(fd, cid, value):
    buf = bytearray(struct.pack("<Ii", cid, int(value)))
    try:
        fcntl.ioctl(fd, VIDIOC_S_CTRL, buf)
    except OSError as exc:
        print(f"  Setting 0x{cid:08X}={value} failed: {err_name(exc)}")
        return False
    got, _ = get_ctrl(fd, cid)
    print(f"  Setting 0x{cid:08X}={value} succeeded (read back {got})")
    return True


def list_all_controls(fd):
    print("=== All Controls Exposed by Driver (NEXT_CTRL Enumeration) ===")
    qid = CTRL_FLAG_NEXT_CTRL
    found = 0
    seen = set()
    while True:
        info = query_ctrl(fd, qid)
        if info is None or info["id"] == 0 or info["id"] in seen:
            break
        seen.add(info["id"])
        found += 1
        value, get_err = get_ctrl(fd, info["id"])
        flags = []
        if info["flags"] & CTRL_FLAG_DISABLED:
            flags.append("disabled")
        if info["flags"] & CTRL_FLAG_READ_ONLY:
            flags.append("read_only")
        if info["flags"] & CTRL_FLAG_INACTIVE:
            flags.append("inactive")
        if info["flags"] & CTRL_FLAG_VOLATILE:
            flags.append("volatile")
        shown = value if get_err is None else f"Read failed ({get_err})"
        ctype = CTRL_TYPE_NAMES.get(info["type"], str(info["type"]))
        print(f"  0x{info['id']:08X}  {info['name'][:28]:<28} "
              f"Current={str(shown):<18} Range={info['min']}..{info['max']}"
              f" Step={info['step']} Default={info['default']}"
              f" Type={ctype} flags=0x{info['flags']:04X}"
              f"{('  [' + ','.join(flags) + ']') if flags else ''}")
        qid = info["id"] | CTRL_FLAG_NEXT_CTRL
    if found == 0:
        print("  (Driver did not expose any controls)")
    print()


# ---------------------------------------------------------------- Formats / FPS

def set_format(fd, width, height, fourcc):
    buf = bytearray(struct.pack(FMT_STRUCT, BUF_TYPE_VIDEO_CAPTURE,
                                width, height, fourcc, FIELD_ANY,
                                0, 0, 0, 0, 0, 0, 0, 0))
    fcntl.ioctl(fd, VIDIOC_S_FMT, buf)
    out = struct.unpack(FMT_STRUCT, buf)
    return {"width": out[1], "height": out[2],
            "fourcc": fourcc_to_str(out[3]), "sizeimage": out[6]}


def get_parm(fd):
    buf = bytearray(struct.pack(PARM_STRUCT, BUF_TYPE_VIDEO_CAPTURE,
                                0, 0, 0, 0, 0, 0, 0))
    fcntl.ioctl(fd, VIDIOC_G_PARM, buf)
    out = struct.unpack(PARM_STRUCT, buf)
    return {"capability": out[1], "numerator": out[3], "denominator": out[4]}


def set_parm(fd, fps):
    current = get_parm(fd)
    can_set = bool(current["capability"] & CAP_TIMEPERFRAME)
    buf = bytearray(struct.pack(PARM_STRUCT, BUF_TYPE_VIDEO_CAPTURE,
                                0, 0, 1, int(fps), 0, 0, 0))
    try:
        fcntl.ioctl(fd, VIDIOC_S_PARM, buf)
    except OSError as exc:
        return can_set, current, None, err_name(exc)
    return can_set, current, get_parm(fd), None


# ---------------------------------------------------------------- Streaming

def measure(fd, size_image, n_frames, n_buffers=4, warmup=15):
    req = bytearray(struct.pack(REQBUFS_STRUCT, n_buffers,
                                BUF_TYPE_VIDEO_CAPTURE, MEMORY_MMAP, 0, 0))
    fcntl.ioctl(fd, VIDIOC_REQBUFS, req)
    n_alloc = struct.unpack(REQBUFS_STRUCT, req)[0]
    if n_alloc < 2:
        raise RuntimeError(f"Driver only allocated {n_alloc} buffers, cannot benchmark")

    maps = []
    for index in range(n_alloc):
        buf = pack_buffer(index)
        fcntl.ioctl(fd, VIDIOC_QUERYBUF, buf)
        info = unpack_buffer(buf)
        offset = info["m"] & 0xFFFFFFFF
        length = info["length"]
        maps.append(mmap.mmap(fd, length, flags=mmap.MAP_SHARED,
                              prot=mmap.PROT_READ | mmap.PROT_WRITE,
                              offset=offset))
        fcntl.ioctl(fd, VIDIOC_QBUF, buf)

    fcntl.ioctl(fd, VIDIOC_STREAMON,
                struct.pack("<i", BUF_TYPE_VIDEO_CAPTURE))

    wall_deltas, drv_deltas, latencies = [], [], []
    n_short = n_timeout = 0
    t_prev = drv_prev = None
    total = warmup + n_frames

    try:
        for i in range(total):
            ready, _, _ = select.select([fd], [], [], 3.0)
            if not ready:
                n_timeout += 1
                continue
            buf = pack_buffer()
            fcntl.ioctl(fd, VIDIOC_DQBUF, buf)
            info = unpack_buffer(buf)
            t_now = time.monotonic()
            drv_now = info["tv_sec"] + info["tv_usec"] / 1e6

            if info["bytesused"] < size_image:
                n_short += 1

            if i >= warmup and t_prev is not None:
                wall_deltas.append((t_now - t_prev) * 1000.0)
                drv_deltas.append((drv_now - drv_prev) * 1000.0)
                latencies.append((t_now - drv_now) * 1000.0)
            t_prev, drv_prev = t_now, drv_now
            fcntl.ioctl(fd, VIDIOC_QBUF, buf)
    finally:
        fcntl.ioctl(fd, VIDIOC_STREAMOFF,
                    struct.pack("<i", BUF_TYPE_VIDEO_CAPTURE))
        for m in maps:
            m.close()

    return {"wall": wall_deltas, "drv": drv_deltas, "lat": latencies,
            "n_short": n_short, "n_timeout": n_timeout, "n_buffers": n_alloc}


def show(stats):
    wall, drv, lat = stats["wall"], stats["drv"], stats["lat"]
    if not wall:
        print("  No valid frames captured.")
        return
    print(f"  Buffer count        : {stats['n_buffers']}")
    print(f"  Wall-clock FPS      : {1000.0 / statistics.mean(wall):6.2f} Hz"
          f"   median interval {statistics.median(wall):6.2f} ms"
          f"   (min {min(wall):.2f} / max {max(wall):.2f})")
    print(f"  Driver timestamp FPS: {1000.0 / statistics.mean(drv):6.2f} Hz"
          f"   median interval {statistics.median(drv):6.2f} ms"
          f"   (min {min(drv):.2f} / max {max(drv):.2f})")
    print(f"  Frame latency (med) : {statistics.median(lat):6.2f} ms")
    print(f"  Undersized frames   : {stats['n_short']}")
    print(f"  select timeouts     : {stats['n_timeout']}")


def main():
    ap = argparse.ArgumentParser(description="V4L2 raw stream benchmark")
    ap.add_argument("device", nargs="?", default="/dev/video0")
    ap.add_argument("--size", default="640x480")
    ap.add_argument("--format", default="YUYV")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--frames", type=int, default=120)
    ap.add_argument("--buffers", type=int, default=4)
    ap.add_argument("--controls-only", action="store_true")
    ap.add_argument("--set", action="append", default=[], metavar="ID=VAL")
    args = ap.parse_args()

    if not os.path.exists(args.device):
        print(f"Device does not exist: {args.device}")
        return 1
    try:
        fd = os.open(args.device, os.O_RDWR)
    except OSError as exc:
        print(f"Open failed: {exc.strerror} (is it in use by another process?)")
        return 1

    try:
        print(f"Device: {args.device}\n")
        list_all_controls(fd)

        for item in args.set:
            if "=" not in item:
                print(f"Ignoring unparseable --set parameter: {item}\n")
                continue
            key, val = item.split("=", 1)
            print(f"Setting {key}:")
            set_ctrl(fd, int(key, 0), int(val, 0))
            print()

        if args.controls_only:
            return 0

        width, height = (int(x) for x in args.size.lower().split("x"))
        fmt = set_format(fd, width, height, fourcc_from_str(args.format))
        print("=== S_FMT Result ===")
        print(f"  Obtained {fmt['fourcc']} {fmt['width']}x{fmt['height']}"
              f"  {fmt['sizeimage']} bytes/frame\n")

        can_set, before, after, parm_err = set_parm(fd, args.fps)
        print(f"=== S_PARM Result (Requested {args.fps} fps) ===")
        print(f"  Driver supports frame rate setting : {can_set}")
        bn, bd = before["numerator"], before["denominator"]
        print(f"  Interval before setting            : {bn}/{bd}"
              f"  → {(bd / bn) if bn else 0:g} fps")
        if parm_err:
            print(f"  S_PARM failed                      : {parm_err}")
        else:
            an, ad = after["numerator"], after["denominator"]
            print(f"  Interval after setting             : {an}/{ad}"
                  f"  → {(ad / an) if an else 0:g} fps")
        print()

        print(f"=== Streaming Benchmark ({args.frames} frames) ===")
        show(measure(fd, fmt["sizeimage"], args.frames,
                     n_buffers=args.buffers))
    finally:
        os.close(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())