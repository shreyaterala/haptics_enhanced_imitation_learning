#!/usr/bin/env python3
"""纯标准库 V4L2 原始取流测速（绕开 OpenCV），并枚举驱动暴露的全部控制项。

用法:
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

# ---- 结构体布局 --------------------------------------------------------
# v4l2_format:      type(4) + pad(4) + v4l2_pix_format(48) + pad = 208
FMT_STRUCT = "<I4x12I152x"
# v4l2_streamparm:  type(4) + raw_data(200) = 204
PARM_STRUCT = "<8I172x"
# v4l2_requestbuffers: 5 字段 = 20
REQBUFS_STRUCT = "<4IB3x"
# v4l2_buffer = 88，timecode 里有 4 个 u8 + userbits[4]
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
    """从 ioctl 编码里取出内核期望的结构体字节数。"""
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
                f"内部错误: {name} 尺寸 {got} 字节，内核期望 {want} 字节。"
                f"请勿运行，否则 ioctl 会越界写内存。")
    n_fields = len(struct.unpack(BUF_STRUCT, b"\x00" * struct.calcsize(BUF_STRUCT)))
    if n_fields != len(BUF_FIELDS):
        raise SystemExit(
            f"内部错误: BUF_STRUCT 解出 {n_fields} 个字段，"
            f"BUF_FIELDS 列了 {len(BUF_FIELDS)} 个。")


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


# ---------------------------------------------------------------- 控制项

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
        print(f"  设置 0x{cid:08X}={value} 失败: {err_name(exc)}")
        return False
    got, _ = get_ctrl(fd, cid)
    print(f"  设置 0x{cid:08X}={value} 成功（回读 {got}）")
    return True


def list_all_controls(fd):
    print("=== 驱动暴露的全部控制项（NEXT_CTRL 枚举）===")
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
        shown = value if get_err is None else f"读取失败({get_err})"
        ctype = CTRL_TYPE_NAMES.get(info["type"], str(info["type"]))
        print(f"  0x{info['id']:08X}  {info['name'][:28]:<28} "
              f"当前={str(shown):<18} 范围={info['min']}..{info['max']}"
              f" 步进={info['step']} 默认={info['default']}"
              f" 类型={ctype} flags=0x{info['flags']:04X}"
              f"{('  [' + ','.join(flags) + ']') if flags else ''}")
        qid = info["id"] | CTRL_FLAG_NEXT_CTRL
    if found == 0:
        print("  （驱动没有暴露任何控制项）")
    print()


# ---------------------------------------------------------------- 格式/帧率

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


# ---------------------------------------------------------------- 取流

def measure(fd, size_image, n_frames, n_buffers=4, warmup=15):
    req = bytearray(struct.pack(REQBUFS_STRUCT, n_buffers,
                                BUF_TYPE_VIDEO_CAPTURE, MEMORY_MMAP, 0, 0))
    fcntl.ioctl(fd, VIDIOC_REQBUFS, req)
    n_alloc = struct.unpack(REQBUFS_STRUCT, req)[0]
    if n_alloc < 2:
        raise RuntimeError(f"驱动只分配了 {n_alloc} 个缓冲，无法测速")

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
        print("  没有采到有效帧。")
        return
    print(f"  缓冲区数量       : {stats['n_buffers']}")
    print(f"  墙上时间帧率     : {1000.0 / statistics.mean(wall):6.2f} Hz"
          f"   间隔中位 {statistics.median(wall):6.2f} ms"
          f"   (最小 {min(wall):.2f} / 最大 {max(wall):.2f})")
    print(f"  驱动时间戳帧率   : {1000.0 / statistics.mean(drv):6.2f} Hz"
          f"   间隔中位 {statistics.median(drv):6.2f} ms"
          f"   (最小 {min(drv):.2f} / 最大 {max(drv):.2f})")
    print(f"  取帧延迟(中位)   : {statistics.median(lat):6.2f} ms")
    print(f"  数据量偏小的帧   : {stats['n_short']}")
    print(f"  select 超时次数  : {stats['n_timeout']}")


def main():
    ap = argparse.ArgumentParser(description="V4L2 原始取流测速")
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
        print(f"设备不存在: {args.device}")
        return 1
    try:
        fd = os.open(args.device, os.O_RDWR)
    except OSError as exc:
        print(f"打开失败: {exc.strerror}（是否被其他进程占用？）")
        return 1

    try:
        print(f"设备: {args.device}\n")
        list_all_controls(fd)

        for item in args.set:
            if "=" not in item:
                print(f"忽略无法解析的 --set 参数: {item}\n")
                continue
            key, val = item.split("=", 1)
            print(f"设置 {key}:")
            set_ctrl(fd, int(key, 0), int(val, 0))
            print()

        if args.controls_only:
            return 0

        width, height = (int(x) for x in args.size.lower().split("x"))
        fmt = set_format(fd, width, height, fourcc_from_str(args.format))
        print("=== S_FMT 结果 ===")
        print(f"  实得 {fmt['fourcc']} {fmt['width']}x{fmt['height']}"
              f"  每帧 {fmt['sizeimage']} 字节\n")

        can_set, before, after, parm_err = set_parm(fd, args.fps)
        print(f"=== S_PARM 结果（请求 {args.fps} fps）===")
        print(f"  驱动声明可设帧率 : {can_set}")
        bn, bd = before["numerator"], before["denominator"]
        print(f"  设置前的间隔     : {bn}/{bd}"
              f"  → {(bd / bn) if bn else 0:g} fps")
        if parm_err:
            print(f"  S_PARM 失败      : {parm_err}")
        else:
            an, ad = after["numerator"], after["denominator"]
            print(f"  设置后的间隔     : {an}/{ad}"
                  f"  → {(ad / an) if an else 0:g} fps")
        print()

        print(f"=== 取流测速（{args.frames} 帧）===")
        show(measure(fd, fmt["sizeimage"], args.frames,
                     n_buffers=args.buffers))
    finally:
        os.close(fd)
    return 0


if __name__ == "__main__":
    sys.exit(main())