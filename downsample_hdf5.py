#!/usr/bin/env python3
"""
方案 1 · 数据端: 把高频 (如 1000 Hz) 采集的 episode 降采样到策略频率 (如 30 Hz)。

核心: 连续信号 (qpos/qvel/action/measured_cf/action_cf...) 先做零相位低通
(Butterworth + filtfilt, 截止 = 0.45 × 目标频率, 防混叠 — 对力信号尤其重要,
直接抽点会把高频接触振动折叠成假的低频分量), 再在均匀目标时间栅格上线性插值。
图像不滤波, 按最近时间戳选帧。

用法:
    python tools/downsample_hdf5.py \
        --input_dir  ./data/dvrk_dataset_1khz \
        --output_dir ./data/dvrk_dataset_30hz \
        --target_fps 30

输入 episode 要求含 /timestamp (秒, 单调)。没有的话用 --source_hz 假定均匀采样。
"""
import argparse
import os
import glob

import numpy as np
import h5py

try:
    from scipy.signal import butter, filtfilt
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# 需要"滤波+重采样"的连续信号键 (存在才处理)
SIGNAL_KEYS = [
    '/observations/qpos',
    '/observations/qvel',
    '/action',
    '/action_cf',
    '/measured_cf',
    '/spatial_cf',
    '/arm_effort',
    '/joint_force',
]
# 图像组: 按最近时间戳选帧
IMAGE_GROUP = '/observations/images'


def lowpass(x, fs, cutoff, order=4):
    """零相位低通; x: (T, D)。scipy 不可用时退化为滑动平均。"""
    if HAS_SCIPY:
        nyq = 0.5 * fs
        wn = min(cutoff / nyq, 0.99)
        b, a = butter(order, wn)
        return filtfilt(b, a, x, axis=0)
    # fallback: 简单滑动平均 (窗口 ≈ fs / (2*cutoff))
    win = max(1, int(round(fs / (2.0 * cutoff))))
    if win <= 1:
        return x
    kernel = np.ones(win) / win
    return np.stack([np.convolve(x[:, d], kernel, mode='same')
                     for d in range(x.shape[1])], axis=1)


def resample_episode(in_path, out_path, target_fps, source_hz=None,
                     filter_order=4, cutoff_ratio=0.45):
    with h5py.File(in_path, 'r') as f:
        if '/timestamp' in f:
            t_src = np.asarray(f['/timestamp'][()], dtype=np.float64)
        else:
            n = f[SIGNAL_KEYS[0]].shape[0] if SIGNAL_KEYS[0] in f else None
            for k in SIGNAL_KEYS:
                if k in f:
                    n = f[k].shape[0]
                    break
            assert source_hz is not None, \
                "episode 没有 /timestamp, 必须给 --source_hz"
            t_src = np.arange(n, dtype=np.float64) / float(source_hz)

        fs = (len(t_src) - 1) / (t_src[-1] - t_src[0])
        duration = t_src[-1] - t_src[0]
        n_out = int(np.floor(duration * target_fps)) + 1
        t_dst = t_src[0] + np.arange(n_out, dtype=np.float64) / target_fps
        cutoff = cutoff_ratio * target_fps

        print(f"  src ≈ {fs:.1f} Hz, {len(t_src)} frames, {duration:.2f}s "
              f"→ dst {target_fps} Hz, {n_out} frames (low-pass {cutoff:.1f} Hz)")

        with h5py.File(out_path, 'w') as g:
            # 属性照抄 + 记录降采样信息
            for k, v in f.attrs.items():
                g.attrs[k] = v
            g.attrs['downsampled_from_hz'] = float(fs)
            g.attrs['downsampled_to_fps']  = float(target_fps)
            g.attrs['antialias_cutoff_hz'] = float(cutoff)

            g.create_dataset('timestamp', data=t_dst)

            # ── 连续信号: 滤波 + 插值 ─────────────────────────────
            for key in SIGNAL_KEYS:
                if key not in f:
                    continue
                x = np.asarray(f[key][()], dtype=np.float64)
                if x.ndim == 1:
                    x = x[:, None]
                T = min(len(x), len(t_src))
                xf = lowpass(x[:T], fs, cutoff, order=filter_order)
                y = np.stack([np.interp(t_dst, t_src[:T], xf[:, d])
                              for d in range(xf.shape[1])], axis=1)
                grp_path, name = key.rsplit('/', 1)
                grp = g.require_group(grp_path) if grp_path else g
                grp.create_dataset(name, data=y.astype(np.float32), dtype='f4')

            # ── 图像: 最近帧 ──────────────────────────────────────
            if IMAGE_GROUP in f:
                img_grp_out = g.require_group(IMAGE_GROUP)
                for cam in f[IMAGE_GROUP]:
                    imgs = f[f'{IMAGE_GROUP}/{cam}']
                    T = min(len(imgs), len(t_src))
                    idx = np.searchsorted(t_src[:T], t_dst)
                    idx = np.clip(idx, 0, T - 1)
                    # searchsorted 给右邻, 和左邻比谁更近
                    left = np.clip(idx - 1, 0, T - 1)
                    pick = np.where(
                        np.abs(t_src[left] - t_dst) <= np.abs(t_src[idx] - t_dst),
                        left, idx)
                    img_grp_out.create_dataset(
                        cam, data=imgs[()][pick],
                        dtype='u1', compression='gzip', compression_opts=2)

            # ── 其余数据集/组原样拷贝 (source_timestamps 等按需保留) ──
            skip = set(SIGNAL_KEYS) | {'/timestamp', IMAGE_GROUP}

            def keep(name):
                full = '/' + name
                return not any(full == s or full.startswith(s + '/') for s in skip)

            def copy_others(name, obj):
                if not keep(name):
                    return
                if isinstance(obj, h5py.Dataset) and ('/' + name) not in g:
                    try:
                        g.create_dataset(name, data=obj[()])
                    except Exception:
                        pass
            f.visititems(copy_others)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input_dir',  required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--target_fps', type=float, default=30.0)
    ap.add_argument('--source_hz',  type=float, default=None,
                    help='episode 无 /timestamp 时假定的均匀采样率 (如 1000)')
    ap.add_argument('--cutoff_ratio', type=float, default=0.45,
                    help='低通截止 = cutoff_ratio × target_fps (< 0.5 防混叠)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, 'episode_*.hdf5')))
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 in {args.input_dir}")
    if not HAS_SCIPY:
        print("[WARN] scipy 不可用, 退化为滑动平均抗混叠 (建议 pip install scipy)")

    for p in files:
        out = os.path.join(args.output_dir, os.path.basename(p))
        print(f"{os.path.basename(p)}:")
        resample_episode(p, out, args.target_fps, source_hz=args.source_hz,
                         cutoff_ratio=args.cutoff_ratio)
    print(f"\nDone. {len(files)} episodes → {args.output_dir}")


if __name__ == '__main__':
    main()
