#!/usr/bin/env python3
"""
Scheme 1 · Data pipeline: downsample high-frequency (e.g. 1000 Hz) recorded episodes to policy frequency (e.g. 30 Hz).

Core: continuous signals (qpos/qvel/action/measured_cf/action_cf...) undergo zero-phase low-pass filtering
(Butterworth + filtfilt, cutoff = 0.45 * target_fps for anti-aliasing — critical for force signals;
direct subsampling would alias high-frequency contact vibrations into false low-frequency components), then linearly interpolate onto uniform target time grid.
Images are not filtered; frames are selected by nearest timestamp.

Usage:
    python tools/downsample_hdf5.py \
        --input_dir  ./data/dvrk_dataset_1khz \
        --output_dir ./data/dvrk_dataset_30hz \
        --target_fps 30

Input episodes must contain /timestamp (seconds, monotonic). If missing, use --source_hz to assume uniform sampling.
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

# Continuous signal keys requiring "filtering + resampling" (processed only if present)
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
# Image groups: select frame by nearest timestamp
IMAGE_GROUP = '/observations/images'


def lowpass(x, fs, cutoff, order=4):
    """Zero-phase low-pass filter; x: (T, D). Falls back to moving average if scipy is unavailable."""
    if HAS_SCIPY:
        nyq = 0.5 * fs
        wn = min(cutoff / nyq, 0.99)
        b, a = butter(order, wn)
        return filtfilt(b, a, x, axis=0)
    # fallback: simple moving average (window ≈ fs / (2*cutoff))
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
                "episode missing /timestamp, --source_hz must be provided"
            t_src = np.arange(n, dtype=np.float64) / float(source_hz)

        fs = (len(t_src) - 1) / (t_src[-1] - t_src[0])
        duration = t_src[-1] - t_src[0]
        n_out = int(np.floor(duration * target_fps)) + 1
        t_dst = t_src[0] + np.arange(n_out, dtype=np.float64) / target_fps
        cutoff = cutoff_ratio * target_fps

        print(f"  src ≈ {fs:.1f} Hz, {len(t_src)} frames, {duration:.2f}s "
              f"→ dst {target_fps} Hz, {n_out} frames (low-pass {cutoff:.1f} Hz)")

        with h5py.File(out_path, 'w') as g:
            # Copy attributes + record downsampling metadata
            for k, v in f.attrs.items():
                g.attrs[k] = v
            g.attrs['downsampled_from_hz'] = float(fs)
            g.attrs['downsampled_to_fps']  = float(target_fps)
            g.attrs['antialias_cutoff_hz'] = float(cutoff)

            g.create_dataset('timestamp', data=t_dst)

            # ── Continuous signals: filter + interpolate ─────────────────────────
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

            # ── Images: nearest frame ──────────────────────────────────────────
            if IMAGE_GROUP in f:
                img_grp_out = g.require_group(IMAGE_GROUP)
                for cam in f[IMAGE_GROUP]:
                    imgs = f[f'{IMAGE_GROUP}/{cam}']
                    T = min(len(imgs), len(t_src))
                    idx = np.searchsorted(t_src[:T], t_dst)
                    idx = np.clip(idx, 0, T - 1)
                    # searchsorted returns right neighbor; compare distance to left neighbor
                    left = np.clip(idx - 1, 0, T - 1)
                    pick = np.where(
                        np.abs(t_src[left] - t_dst) <= np.abs(t_src[idx] - t_dst),
                        left, idx)
                    img_grp_out.create_dataset(
                        cam, data=imgs[()][pick],
                        dtype='u1', compression='gzip', compression_opts=2)

            # ── Copy remaining datasets/groups as-is (e.g., source_timestamps) ──
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
                    help='Assumed uniform sampling rate when episode lacks /timestamp (e.g. 1000)')
    ap.add_argument('--cutoff_ratio', type=float, default=0.45,
                    help='Low-pass cutoff = cutoff_ratio * target_fps (< 0.5 for anti-aliasing)')
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.input_dir, 'episode_*.hdf5')))
    if not files:
        raise FileNotFoundError(f"No episode_*.hdf5 in {args.input_dir}")
    if not HAS_SCIPY:
        print("[WARN] scipy unavailable, falling back to moving average anti-aliasing (recommended: pip install scipy)")

    for p in files:
        out = os.path.join(args.output_dir, os.path.basename(p))
        print(f"{os.path.basename(p)}:")
        resample_episode(p, out, args.target_fps, source_hz=args.source_hz,
                         cutoff_ratio=args.cutoff_ratio)
    print(f"\nDone. {len(files)} episodes → {args.output_dir}")


if __name__ == '__main__':
    main()
