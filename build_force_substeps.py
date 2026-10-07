#!/usr/bin/env python3
"""
方案 2 · 数据端: 把 /action_cf_stream (1 kHz force_goal 原始流) 切成与相机帧
对齐的子步标签 /action_cf_hi, 形状 (T, K, 6):

    帧 i 的第 j 个子步时刻:  t_ij = t_i + (j + 0.5) / K * (t_{i+1} - t_i)

即每个策略步 (帧间隔 ≈ 1/fps) 内均匀取 K 个点, 各通道对原始流线性插值。
训练时该 (K, 6) 块被展平进 action:  action_t = [q_cmd(7) | f_cmd 展平(K*6)]。

用法:
    python tools/build_force_substeps.py \
        --dataset_dir ./data/dvrk_dataset_bilateral \
        --substeps 33            # 30 Hz 策略 → K = 1000/30 ≈ 33
                                 # 3  Hz 策略 → K = 333

原地写入 (在每个 episode HDF5 里新增 /action_cf_hi 与 attrs['force_substeps'])。
"""
import argparse
import os
import glob

import numpy as np
import h5py


def build_substeps_for_episode(path, K, overwrite=False):
    with h5py.File(path, 'a') as f:
        if '/action_cf_hi' in f:
            if not overwrite:
                print(f"  跳过 (已有 /action_cf_hi, 用 --overwrite 重建)")
                return False
            del f['/action_cf_hi']

        assert '/timestamp' in f, f"{path}: 缺少 /timestamp"
        t_frames = np.asarray(f['/timestamp'][()], dtype=np.float64)
        T = len(t_frames)

        if '/action_cf_stream' in f and f['/action_cf_stream'].shape[0] > 1:
            stream = np.asarray(f['/action_cf_stream'][()], dtype=np.float64)
            t_s, w_s = stream[:, 0], stream[:, 1:7]
            # 保证单调 (乱序消息按时间排序)
            order = np.argsort(t_s)
            t_s, w_s = t_s[order], w_s[order]
            src_desc = f"stream ({len(t_s)} msgs, " \
                       f"~{(len(t_s)-1)/max(t_s[-1]-t_s[0],1e-9):.0f} Hz)"
        elif '/action_cf' in f:
            # 退化: 只有逐帧标签 → 子步 = 帧间线性插值 (没有帧内高频信息,
            # 等价于方案 1 的线性上采样, 仅保证管线可跑通)
            w_frames = np.asarray(f['/action_cf'][()], dtype=np.float64)[:T]
            t_s, w_s = t_frames[:len(w_frames)], w_frames
            src_desc = "per-frame /action_cf (⚠️ 无帧内高频信息, 仅插值)"
        else:
            raise KeyError(f"{path}: 既无 /action_cf_stream 也无 /action_cf")

        # 帧间隔 (最后一帧用中位数间隔外推)
        dt = np.diff(t_frames)
        dt_last = np.median(dt) if len(dt) > 0 else 1.0 / 30.0
        t_next = np.concatenate([t_frames[1:], [t_frames[-1] + dt_last]])

        # 子步时刻 (T, K)
        j = (np.arange(K, dtype=np.float64) + 0.5) / K
        t_sub = t_frames[:, None] + j[None, :] * (t_next - t_frames)[:, None]

        # 逐通道插值 → (T, K, 6)
        hi = np.stack(
            [np.interp(t_sub.ravel(), t_s, w_s[:, d]).reshape(T, K)
             for d in range(6)], axis=-1).astype(np.float32)

        f.create_dataset('action_cf_hi', data=hi, dtype='f4')
        f.attrs['force_substeps'] = int(K)
        print(f"  /action_cf_hi {hi.shape}  来源: {src_desc}")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--substeps', type=int, required=True,
                    help='每个策略步的子步数 K = round(1000 / policy_fps)')
    ap.add_argument('--overwrite', action='store_true')
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dataset_dir, 'episode_*.hdf5')))
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 in {args.dataset_dir}")

    n_done = 0
    for p in files:
        print(f"{os.path.basename(p)}:")
        if build_substeps_for_episode(p, args.substeps, overwrite=args.overwrite):
            n_done += 1
    print(f"\nDone. {n_done}/{len(files)} episodes updated (K={args.substeps}).")


if __name__ == '__main__':
    main()
