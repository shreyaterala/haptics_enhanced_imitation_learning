#!/usr/bin/env python3
"""
Scheme 2 · Data pipeline: slice /action_cf_stream (1 kHz raw force_goal stream) into
substep labels /action_cf_hi aligned with camera frames, shape (T, K, 6):

    Substep j timestamp for frame i:  t_ij = t_i + (j + 0.5) / K * (t_{i+1} - t_i)

Uniformly sample K points within each policy step (frame interval ≈ 1/fps), linearly interpolating channels from raw stream.
During training, this (K, 6) block is flattened into action: action_t = [q_cmd(7) | f_cmd flattened (K*6)].

Usage:
    python tools/build_force_substeps.py \
        --dataset_dir ./data/dvrk_dataset_bilateral \
        --substeps 33            # 30 Hz policy → K = 1000/30 ≈ 33
                                 # 3  Hz policy → K = 333

In-place write (adds /action_cf_hi and attrs['force_substeps'] to each episode HDF5).
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
                print(f"  Skipping (already has /action_cf_hi, use --overwrite to rebuild)")
                return False
            del f['/action_cf_hi']

        assert '/timestamp' in f, f"{path}: missing /timestamp"
        t_frames = np.asarray(f['/timestamp'][()], dtype=np.float64)
        T = len(t_frames)

        if '/action_cf_stream' in f and f['/action_cf_stream'].shape[0] > 1:
            stream = np.asarray(f['/action_cf_stream'][()], dtype=np.float64)
            t_s, w_s = stream[:, 0], stream[:, 1:7]
            # Ensure monotonicity (sort out-of-order messages by time)
            order = np.argsort(t_s)
            t_s, w_s = t_s[order], w_s[order]
            src_desc = f"stream ({len(t_s)} msgs, " \
                       f"~{(len(t_s)-1)/max(t_s[-1]-t_s[0],1e-9):.0f} Hz)"
        elif '/action_cf' in f:
            # Fallback: only per-frame labels available -> substeps = linear interpolation between frames
            # (no intra-frame high-frequency information, equivalent to Scheme 1 linear upsampling, ensures pipeline runnable)
            w_frames = np.asarray(f['/action_cf'][()], dtype=np.float64)[:T]
            t_s, w_s = t_frames[:len(w_frames)], w_frames
            src_desc = "per-frame /action_cf (⚠️ no intra-frame high-frequency info, interpolation only)"
        else:
            raise KeyError(f"{path}: neither /action_cf_stream nor /action_cf found")

        # Frame intervals (last frame extrapolated using median interval)
        dt = np.diff(t_frames)
        dt_last = np.median(dt) if len(dt) > 0 else 1.0 / 30.0
        t_next = np.concatenate([t_frames[1:], [t_frames[-1] + dt_last]])

        # Substep timestamps (T, K)
        j = (np.arange(K, dtype=np.float64) + 0.5) / K
        t_sub = t_frames[:, None] + j[None, :] * (t_next - t_frames)[:, None]

        # Channel-wise interpolation -> (T, K, 6)
        hi = np.stack(
            [np.interp(t_sub.ravel(), t_s, w_s[:, d]).reshape(T, K)
             for d in range(6)], axis=-1).astype(np.float32)

        f.create_dataset('action_cf_hi', data=hi, dtype='f4')
        f.attrs['force_substeps'] = int(K)
        print(f"  /action_cf_hi {hi.shape}  source: {src_desc}")
        return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset_dir', required=True)
    ap.add_argument('--substeps', type=int, required=True,
                    help='Number of substeps K per policy step = round(1000 / policy_fps)')
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
