# ACT-dVRK-Bilateral: Enabling ACT to Serve as Bilateral Teleoperation

Modified based on `act_dvrk_jf`. Core idea (Bi-ACT style): **The policy replaces the MTM / human operator**,
simultaneously predicting position commands and force goals for the follower arm at each step. These commands are executed by low-level dual-channel control (`servo_jp` + `servo_cf`),
reproducing the dual position + force channels sent to the PSM in the `dvrk_bila_teleop.py` FOLLOWING loop without needing a physical master arm (velocity channel can be obtained by differentiating predicted action chunks, see below).

## Inputs / Outputs

| Direction | Signal | Dimension | Description |
|---|---|---|---|
| Input token | qpos | state_dim (7) | PSM arm[6] + jaw[1], same as jf |
| Input token | **qvel** | state_dim (7) | **New**, existing `/observations/qvel` in HDF5 |
| Input token | cf | CF_DIM (6) | External force/wrench (sensor or NN estimate), same as jf |
| Input | images | — | Same as jf |
| Output | **action** | state_dim + action_cf_dim (7+6=13) | **Per step = [q_cmd | f_cmd]** |

Decoder additional tokens are fixed to 4: `latent / qpos / qvel / force`
(`additional_pos_embed = nn.Embedding(4, hidden)`).

Supervision labels for `f_cmd` prioritize `/action_cf` (the actual `force_goal` issued during teleop);
if unavailable in the dataset, it falls back to `/measured_cf` (approximate supervision; re-collecting data is recommended).

Loss: `L = L1(q_cmd) + force_loss_weight * L1(f_cmd) + kl_weight * KL`,
`l1` and `l1_force` are logged separately in training logs.

## Data Collection (New force_goal Pipeline)

1. Perform teleoperated demonstrations using `bilateral_teleoperation/dvrk_bila_teleop_bilateral.py`
   (which publishes `force_goal` to `/teleop/force_goal` on every FOLLOWING cycle).
2. During recording, `record_dvrk_episodes.py` automatically subscribes to this topic and saves it as `/action_cf`
   (disable via `--force_goal_topic none`; missing frames are held from the previous frame to maintain alignment with action).

```bash
python data_recording/record_dvrk_episodes.py \
    --output_dir ./data/dvrk_dataset_bilateral \
    --seconds_per_episode 10 --fps 20 \
    --wrist_cameras cam1 --force_sensor
```

## Training

```bash
python imitate_episodes.py \
    --task_name dvrk_retraction_bilateral \
    --ckpt_dir ./checkpoints/dvrk_bilateral \
    --policy_class ACT --batch_size 8 --seed 0 \
    --num_epochs 5000 --lr 1e-5 \
    --state_dim 7 --chunk_size 100 \
    --hidden_dim 512 --dim_feedforward 3200 --kl_weight 10 \
    --action_cf_dim 6 --force_loss_weight 1.0 \
    --temporal_agg
```

## Deployment (Dual Position + Force Channels)

```bash
python run_dvrk_act.py \
    --ckpt_dir ./checkpoints/dvrk_bilateral \
    --task_name dvrk_retraction_bilateral \
    --chunk_size 100 --fps 20 --temporal_agg \
    --send_force --force_gain 0.5 --force_cmd_clip 5.0 --zero_torque_cmd
```

- Without `--send_force`, `f_cmd` is only logged and not commanded to hardware (recommended for initial test runs to verify predicted forces).
- `--force_gain` can be ramped up gradually from 0.5 to 1.0; `--force_cmd_clip` specifies safety clipping per component (N / N*m).
- `--zero_torque_cmd` zeros out the torque components, consistent with `cf[3:6] *= 0` in `dvrk_bila_teleop.py`.
- Temporal aggregation performs exponential weighting on the full 13-dim action (including `f_cmd`).

## Comparison with jf Variant

| File | Changes |
|---|---|
| `detr/models/detr_vae.py` | Decoupled `action_dim` and `state_dim`; `action_head`/`encoder_action_proj` use `action_dim`; added `input_proj_robot_vel`; 4 additional tokens |
| `detr/models/transformer.py` | Forward pass accepts `qvel_input`, token stack `[latent, qpos, qvel, force]` |
| `detr/main.py` | Added `--cf_dim / --action_cf_dim / --action_dim` |
| `policy.py` | Signature `(qpos, qvel, cf, image, ...)`; loss split into `l1` / `l1_force` |
| `utils.py` | Loads `qvel`; action = `/action` ⊕ `/action_cf`; stats include `qvel_mean/std`, `action_cf_dim`, `force_label_key`; std floor 1e-2 on `f_cmd` dims |
| `imitate_episodes.py` | New CLI parameters and config forwarding; forward_pass returns 6-tuple; eval loop adaptations |
| `run_dvrk_act.py` | Fixed calls previously incompatible with Scheme B; independent tokens for qpos/qvel/cf; splits `[q_cmd\|f_cmd]`; `ArmCustom.body.servo_cf` sends wrench; added `--send_force`, etc. |
| `data_recording/record_dvrk_episodes.py` | Subscribes to `/teleop/force_goal` -> stores `/action_cf` |

## Two Schemes for Handling Frequency (Hz) Mismatch

At policy frequency `f_policy`, each step corresponds to `K = 1000 / f_policy` 1 kHz substeps (30 Hz -> K≈33, 3 Hz -> K=333).

### Scheme 1 — Downsampled Training + Deployment Interpolation Upsampling

Data pipeline (1 kHz recording -> anti-aliasing low-pass filter -> 30 Hz):

```bash
python tools/downsample_hdf5.py \
    --input_dir ./data/dvrk_dataset_1khz \
    --output_dir ./data/dvrk_dataset_30hz \
    --target_fps 30
```

Deployment pipeline (30 Hz policy output, executor thread interpolates and dispatches at 1 kHz;
q linear interpolation, f linear interpolation or zero-order hold, optional servo_jv velocity feedforward = interpolation slope,
which also recreates the teleop servo_cv channel):

```bash
python run_dvrk_act.py ... --send_force \
    --interp_hz 1000 --interp_mode linear --send_jv
```

### Scheme 2 — Preserve 1 kHz Supervision: Flatten K Force Substeps per Step

Action becomes `[q_cmd(7) | f_cmd(K×6)]` (when K=33, action_dim = 205).
The model directly predicts the entire intra-frame force profile for each chunk step, preserving high-frequency contact transients.

Data pipeline (recording script automatically saves raw 1 kHz force_goal stream `/action_cf_stream`,
then slices into frame-aligned substeps):

```bash
python tools/build_force_substeps.py \
    --dataset_dir ./data/dvrk_dataset_bilateral --substeps 33
```

Training:

```bash
python imitate_episodes.py ... --force_substeps 33
```

Deployment (position remains interpolated, force is played back substep-by-substep via `profile[floor(alpha*K)]`):

```bash
python run_dvrk_act.py ... --send_force --interp_hz 1000
# force_substeps is automatically loaded from stats/train_config
```

### Comparison of the Two Schemes

| | Scheme 1 (Downsampling + Interpolation) | Scheme 2 (Flattened K Substeps) |
|---|---|---|
| Force label bandwidth | <= 0.45 x fps (~13.5 Hz) | Near 500 Hz (Nyquist) |
| action_dim | 13 | 7 + 6K (K=33 -> 205) |
| High-frequency contact transients | Smoothed out by low-pass filter; deployment uses linear transition | Model directly predicts intra-frame profile |
| Training difficulty / data demands | Low | High (large output dimension; recommend lowering force_loss_weight, e.g. 0.3~0.5) |
| Implementation risk | Low | Medium (profile prediction jitter needs observation; dry-run first to inspect f profile) |

It is recommended to validate Scheme 1 first as a baseline, and then evaluate Scheme 2 to compare force tracking quality.

## Notes

- **Executor measured frequency**: Python threads are subject to GIL and scheduling constraints, typically measuring 300~800 Hz;
  the dVRK middleware itself runs at ~1.5 kHz and holds the latest setpoint, so the primary goal is smoothness rather than strictly 1 kHz.
  Measured frequency is printed upon exit; if significantly insufficient, consider lowering `--interp_hz` or pushing interpolation down into C++ middleware.
- **Anti-aliasing**: In Scheme 1, naive decimation (without filtering) aliases high-frequency contact vibrations into false low-frequency force signals.
  Always process data via `downsample_hdf5.py` (synthetic 80 Hz vibration test: filtered residual 0.004 vs. naive decimation 0.42).
- Legacy datasets (containing only `/measured_cf` and no `/action_cf`) can be trained directly; `utils.py` will
  automatically fall back to `measured_cf` as the force label and log a warning. However, semantically it represents "measured external force"
  rather than the "commanded force goal", so re-collecting data with the new pipeline is recommended.
