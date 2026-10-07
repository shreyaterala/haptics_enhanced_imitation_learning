# ACT-dVRK-Bilateral: 让 ACT 承担 bilateral teleoperation 的角色

基于 `act_dvrk_jf` 修改。核心思想(Bi-ACT 式):**策略取代 MTM/操作者**,
每步同时输出从手的位置指令与力目标,由底层双通道控制(servo_jp + servo_cf)执行,
从而在没有主手的情况下复现 `dvrk_bila_teleop.py` FOLLOWING 循环里发往 PSM 的
position + force 两条通道(velocity 通道可由预测 chunk 差分得到,见下)。

## 输入 / 输出

| 方向 | 信号 | 维度 | 说明 |
|---|---|---|---|
| 输入 token | qpos | state_dim (7) | PSM arm[6]+jaw[1],同 jf |
| 输入 token | **qvel** | state_dim (7) | **新增**,HDF5 里已有的 `/observations/qvel` |
| 输入 token | cf | CF_DIM (6) | 外力 (传感器或 NN 估计),同 jf |
| 输入 | images | — | 同 jf |
| 输出 | **action** | state_dim + action_cf_dim (7+6=13) | **每步 = [q_cmd \| f_cmd]** |

decoder 附加 token 固定为 4 个:latent / qpos / qvel / force
(`additional_pos_embed = nn.Embedding(4, hidden)`)。

f_cmd 的监督标签优先取 `/action_cf`(teleop 实际下发的 force_goal);
数据里没有时退化用 `/measured_cf`(近似监督,建议重新采集)。

损失:`L = L1(q_cmd) + force_loss_weight * L1(f_cmd) + kl_weight * KL`,
训练日志中分别记录 `l1` 与 `l1_force`。

## 数据采集(新增 force_goal 链路)

1. 用 `bilateral_teleoperation/dvrk_bila_teleop_bilateral.py` 做演示
   (它在每个 FOLLOWING 周期把 force_goal 发布到 `/teleop/force_goal`)。
2. 录制时 `record_dvrk_episodes.py` 自动订阅该话题并存为 `/action_cf`
   (可用 `--force_goal_topic none` 关闭;缺帧用上一帧补齐保证与 action 对齐)。

```bash
python data_recording/record_dvrk_episodes.py \
    --output_dir ./data/dvrk_dataset_bilateral \
    --seconds_per_episode 10 --fps 20 \
    --wrist_cameras cam1 --force_sensor
```

## 训练

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

## 部署(位置 + 力双通道)

```bash
python run_dvrk_act.py \
    --ckpt_dir ./checkpoints/dvrk_bilateral \
    --task_name dvrk_retraction_bilateral \
    --chunk_size 100 --fps 20 --temporal_agg \
    --send_force --force_gain 0.5 --force_cmd_clip 5.0 --zero_torque_cmd
```

- 不加 `--send_force` 时 f_cmd 只打印不下发(建议第一次先这样跑,观察预测的力是否合理)。
- `--force_gain` 从 0.5 逐步升到 1.0;`--force_cmd_clip` 是每分量安全限幅 (N / N·m)。
- `--zero_torque_cmd` 将力矩分量清零,与 `dvrk_bila_teleop.py` 中 `cf[3:6] *= 0` 行为一致。
- temporal aggregation 对完整的 13 维动作(含 f_cmd)一起做指数加权。

## 与 jf 变体的差异一览

| 文件 | 改动 |
|---|---|
| `detr/models/detr_vae.py` | `action_dim` 与 `state_dim` 解耦;`action_head`/`encoder_action_proj` 用 action_dim;新增 `input_proj_robot_vel`;附加 token 4 个 |
| `detr/models/transformer.py` | forward 接受 `qvel_input`,token 栈 [latent, qpos, qvel, force] |
| `detr/main.py` | 新增 `--cf_dim / --action_cf_dim / --action_dim` |
| `policy.py` | 签名 `(qpos, qvel, cf, image, ...)`;损失拆分 `l1` / `l1_force` |
| `utils.py` | 加载 qvel;action = `/action` ⊕ `/action_cf`;stats 增加 qvel_mean/std、action_cf_dim、force_label_key;f_cmd 维 std 下限 1e-2 |
| `imitate_episodes.py` | 新参数与 config 传递;forward_pass 6 元组;eval 循环适配 |
| `run_dvrk_act.py` | 修复原本与方案 B 不兼容的调用;qpos/qvel/cf 独立 token;拆分 [q_cmd\|f_cmd];`ArmCustom.body.servo_cf` 下发力;新增 `--send_force` 等参数 |
| `data_recording/record_dvrk_episodes.py` | 订阅 `/teleop/force_goal` → 存 `/action_cf` |

## Hz 不匹配的两种方案

策略步频 f_policy 下每步对应 K = 1000 / f_policy 个 1 kHz 子步(30 Hz → K≈33,3 Hz → K=333)。

### 方案 1 — 降采样训练 + 部署插值上采样

数据端(1 kHz 采集 → 抗混叠低通 → 30 Hz):

```bash
python tools/downsample_hdf5.py \
    --input_dir ./data/dvrk_dataset_1khz \
    --output_dir ./data/dvrk_dataset_30hz \
    --target_fps 30
```

部署端(30 Hz 策略输出,executor 线程 1 kHz 插值下发;q 线性插值、
f 线性插值或零阶保持、可选 servo_jv 速度前馈 = 插值斜率,顺带把
teleop 的 servo_cv 通道也复原了):

```bash
python run_dvrk_act.py ... --send_force \
    --interp_hz 1000 --interp_mode linear --send_jv
```

### 方案 2 — 保持 1 kHz 监督:每步动作展平 K 个力子步

动作变为 `[q_cmd(7) | f_cmd(K×6)]`(K=33 时 action_dim = 205),
模型每个 chunk 步直接预测整段帧内力剖面,保留接触瞬态等高频信息。

数据端(录制脚本已自动保存 1 kHz force_goal 原始流 `/action_cf_stream`,
再切成帧对齐子步):

```bash
python tools/build_force_substeps.py \
    --dataset_dir ./data/dvrk_dataset_bilateral --substeps 33
```

训练:

```bash
python imitate_episodes.py ... --force_substeps 33
```

部署(位置仍插值,力按 profile[⌊α·K⌋] 逐子步回放):

```bash
python run_dvrk_act.py ... --send_force --interp_hz 1000
# force_substeps 自动从 stats/train_config 读取
```

### 两种方案对比

| | 方案 1 (降采样+插值) | 方案 2 (K 子步展平) |
|---|---|---|
| 力标签带宽 | ≤ 0.45×fps (≈13.5 Hz) | 接近 500 Hz (Nyquist) |
| action_dim | 13 | 7 + 6K (K=33 → 205) |
| 高频接触瞬态 | 被低通抹掉,部署端只能线性过渡 | 模型可直接输出帧内剖面 |
| 训练难度/数据需求 | 低 | 高 (输出维度大, 建议 force_loss_weight 调低如 0.3~0.5 起步) |
| 实现风险 | 低 | 中 (剖面预测抖动需观察; 可先 dry-run 看 f 剖面) |

建议先跑通方案 1 作为 baseline,再上方案 2 对比力跟踪质量。

## 备注

- **executor 实测频率**:Python 线程受 GIL/调度限制,实测可能 300~800 Hz;
  dVRK 中间件本身 ~1.5 kHz 并保持最近 setpoint,重点是平滑而非严格 1 kHz。
  退出时会打印实测频率;若明显不足可降低 --interp_hz 或把插值下沉到 C++ 中间件。
- **抗混叠**:方案 1 直接抽点(不滤波)会把高频接触振动折叠成假的低频力信号,
  务必走 downsample_hdf5.py(合成 80 Hz 振动测试:滤波后残差 0.004 vs 直接抽点 0.42)。
- 旧数据(只有 `/measured_cf`、没有 `/action_cf`)可以直接训练,utils 会
  自动退化用 measured_cf 作为力标签并打印警告,但语义上它是"测得的外力"
  而非"应下发的力目标",建议用新链路重新采集。
