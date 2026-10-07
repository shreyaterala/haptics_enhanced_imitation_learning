import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed


def _find_episode(dataset_dir, episode_id):
    dataset_path_padded = os.path.join(dataset_dir, f'episode_{episode_id:03d}.hdf5')
    dataset_path_nopad  = os.path.join(dataset_dir, f'episode_{episode_id}.hdf5')
    if os.path.exists(dataset_path_padded):
        return dataset_path_padded
    elif os.path.exists(dataset_path_nopad):
        return dataset_path_nopad
    raise FileNotFoundError(
        f"Episode file not found for index {episode_id}.\n"
        f"  Tried: {dataset_path_padded}\n"
        f"  Tried: {dataset_path_nopad}"
    )


def _force_label_key(root):
    """
    动作里 f_cmd 的标签来源(按优先级):
      1. /action_cf     — teleop 采集时记录的 force_goal(推荐,即 servo_cf 实际发送的 wrench)
      2. /measured_cf   — 外力传感器/估计,退而求其次(近似监督)
    """
    if '/action_cf' in root:
        return '/action_cf'
    return '/measured_cf'


class EpisodicDataset(torch.utils.data.Dataset):
    """
    Bilateral 版:
      观测: qpos (state_dim), qvel (state_dim), cf (cf_dim), images
      动作: [q_cmd (state_dim) | f_cmd (action_cf_dim)]  → action_dim
    """
    def __init__(self, episode_ids, dataset_dir, camera_names, norm_stats,
                 state_dim, chunk_size, cf_dim, action_cf_dim, cf_clip=None,
                 force_substeps=0):
        super(EpisodicDataset).__init__()
        self.episode_ids   = episode_ids
        self.dataset_dir   = dataset_dir
        self.camera_names  = camera_names
        self.norm_stats    = norm_stats
        self.state_dim     = state_dim
        self.chunk_size    = chunk_size
        self.cf_dim        = cf_dim
        self.action_cf_dim = action_cf_dim
        # 方案2: K>0 时每步动作的力部分 = K 个子步展平 (K * action_cf_dim)
        self.force_substeps = int(force_substeps)
        f_total = action_cf_dim * max(self.force_substeps, 1)
        self.action_dim    = state_dim + f_total

        if cf_clip is not None:
            self.cf_clip = float(cf_clip)
        elif "cf_clip" in norm_stats:
            self.cf_clip = float(norm_stats["cf_clip"])
        else:
            self.cf_clip = None

        self.clip_hit_count = 0
        self.total_count    = 0

        self.is_sim = None
        self.__getitem__(0)

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        sample_full_episode = False

        episode_id   = self.episode_ids[index]
        dataset_path = _find_episode(self.dataset_dir, episode_id)

        with h5py.File(dataset_path, 'r') as root:
            is_sim   = root.attrs['sim']
            if self.force_substeps > 0:
                fkey = '/action_cf_hi'
                assert fkey in root, \
                    (f"{dataset_path}: force_substeps={self.force_substeps} 需要 "
                     f"/action_cf_hi (先跑 tools/build_force_substeps.py)")
                assert root[fkey].shape[1] >= self.force_substeps, \
                    f"/action_cf_hi K={root[fkey].shape[1]} < {self.force_substeps}"
            else:
                fkey = _force_label_key(root)
            T_action = root['/action'].shape[0]
            T_qpos   = root['/observations/qpos'].shape[0]
            T_cf     = root['/measured_cf'].shape[0]
            T_fcmd   = root[fkey].shape[0]
            episode_len = min(T_action, T_qpos, T_cf, T_fcmd)

            if sample_full_episode:
                start_ts = 0
            else:
                start_ts = np.random.choice(episode_len)

            qpos = root['/observations/qpos'][start_ts, :self.state_dim]
            qvel = root['/observations/qvel'][start_ts, :self.state_dim]
            cf   = root['/measured_cf'][start_ts, :self.cf_dim]

            image_dict = dict()
            for cam_name in self.camera_names:
                image_dict[cam_name] = root[f'/observations/images/{cam_name}'][start_ts]

            # 动作序列: q_cmd 与 f_cmd 使用相同的起点索引,保证逐时刻对齐
            if is_sim:
                a_start = start_ts
            else:
                a_start = max(0, start_ts - 1)
            q_cmd  = root['/action'][a_start:episode_len, :self.state_dim]
            if self.force_substeps > 0:
                # (T', K, af_dim) → 展平 (T', K*af_dim)
                f_hi  = root[fkey][a_start:episode_len,
                                   :self.force_substeps, :self.action_cf_dim]
                f_cmd = f_hi.reshape(f_hi.shape[0], -1)
            else:
                f_cmd = root[fkey][a_start:episode_len, :self.action_cf_dim]
            action = np.concatenate([q_cmd, f_cmd], axis=-1)   # (T', action_dim)
            action_len = episode_len - a_start

        self.is_sim = is_sim

        valid_len     = min(action_len, self.chunk_size)
        padded_action = np.zeros((self.chunk_size, self.action_dim), dtype=np.float32)
        padded_action[:valid_len] = action[:valid_len]
        is_pad        = np.zeros(self.chunk_size)
        if valid_len < self.chunk_size:
            is_pad[valid_len:] = 1

        all_cam_images = []
        for cam_name in self.camera_names:
            all_cam_images.append(image_dict[cam_name])
        all_cam_images = np.stack(all_cam_images, axis=0)

        image_data  = torch.from_numpy(all_cam_images)
        qpos_data   = torch.from_numpy(qpos).float()
        qvel_data   = torch.from_numpy(qvel).float()
        cf_data     = torch.from_numpy(cf).float()
        action_data = torch.from_numpy(padded_action).float()
        is_pad      = torch.from_numpy(is_pad).bool()

        image_data = torch.einsum('k h w c -> k c h w', image_data)

        # ── 归一化 ───────────────────────────────────────────────────
        image_data  = image_data / 255.0
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]
        qpos_data   = (qpos_data   - self.norm_stats["qpos_mean"])   / self.norm_stats["qpos_std"]
        qvel_data   = (qvel_data   - self.norm_stats["qvel_mean"])   / self.norm_stats["qvel_std"]
        cf_data     = (cf_data     - self.norm_stats["cf_mean"])     / self.norm_stats["cf_std"]

        # ── cf clip(只作用于观测端的力,不动动作标签)─────────────
        if self.cf_clip is not None:
            pre_clip = cf_data
            cf_data  = torch.clamp(cf_data, -self.cf_clip, self.cf_clip)
            if (pre_clip != cf_data).any().item():
                self.clip_hit_count += 1
            self.total_count += 1

        return image_data, qpos_data, qvel_data, cf_data, action_data, is_pad


def get_norm_stats(dataset_dir, num_episodes, state_dim, cf_dim, action_cf_dim,
                   cf_clip=3.0, force_substeps=0):
    all_qpos_data   = []
    all_qvel_data   = []
    all_action_data = []   # 只放 q_cmd 部分; f_cmd 通道统计单独算再平铺
    all_fcmd_data   = []   # (N, af_dim) 力通道样本 (substeps 时展开到子步级)
    all_cf_data     = []

    force_key_used = None
    for episode_idx in range(num_episodes):
        dataset_path = _find_episode(dataset_dir, episode_idx)

        with h5py.File(dataset_path, 'r') as root:
            if force_substeps > 0:
                fkey = '/action_cf_hi'
                assert fkey in root, \
                    f"{dataset_path}: 缺 /action_cf_hi (先跑 tools/build_force_substeps.py)"
            else:
                fkey = _force_label_key(root)
            if force_key_used is None:
                force_key_used = fkey
                print(f'[get_norm_stats] f_cmd label source: {fkey}'
                      + ('' if fkey in ('/action_cf', '/action_cf_hi')
                         else '  (⚠️ 没找到 /action_cf,退化用 measured_cf 作为力标签)'))
            qpos   = root['/observations/qpos'][()]
            qvel   = root['/observations/qvel'][()]
            action = root['/action'][()]
            cf     = root['/measured_cf'][()]
            f_cmd  = root[fkey][()]
        T = min(len(qpos), len(qvel), len(action), len(cf), len(f_cmd))
        qpos, qvel, action, cf, f_cmd = qpos[:T], qvel[:T], action[:T], cf[:T], f_cmd[:T]

        if force_substeps > 0:
            # (T, K, af) → 子步级样本 (T*K, af): 通道统计
            f_samples = f_cmd[:, :force_substeps, :action_cf_dim].reshape(-1, action_cf_dim)
        else:
            f_samples = f_cmd[:, :action_cf_dim]

        all_qpos_data.append(torch.from_numpy(qpos[:, :state_dim]).float())
        all_qvel_data.append(torch.from_numpy(qvel[:, :state_dim]).float())
        all_action_data.append(torch.from_numpy(action[:, :state_dim]).float())
        all_fcmd_data.append(torch.from_numpy(np.ascontiguousarray(f_samples)).float())
        all_cf_data.append(torch.from_numpy(cf[:, :cf_dim]).float())

    all_qpos_data   = torch.cat(all_qpos_data,   dim=0)
    all_qvel_data   = torch.cat(all_qvel_data,   dim=0)
    all_action_data = torch.cat(all_action_data, dim=0)
    all_fcmd_data   = torch.cat(all_fcmd_data,   dim=0)
    all_cf_data     = torch.cat(all_cf_data,     dim=0)

    # q_cmd 部分: 沿用 0.1 std 下限
    qcmd_mean = all_action_data.mean(dim=0, keepdim=True)
    qcmd_std  = torch.clip(all_action_data.std(dim=0, keepdim=True), 0.1, np.inf)

    # f_cmd 部分: 通道级统计, 下限 1e-2 (力幅值可能远小于 0.1 N)
    fcmd_mean = all_fcmd_data.mean(dim=0, keepdim=True)
    fcmd_std  = torch.clip(all_fcmd_data.std(dim=0, keepdim=True), 1e-2, np.inf)

    # substeps 时把通道 stats 平铺 K 次, 保证每个子步同一通道用同一归一化
    K_rep = max(int(force_substeps), 1)
    action_mean = torch.cat([qcmd_mean, fcmd_mean.repeat(1, K_rep)], dim=1)
    action_std  = torch.cat([qcmd_std,  fcmd_std.repeat(1, K_rep)],  dim=1)

    qpos_mean = all_qpos_data.mean(dim=0, keepdim=True)
    qpos_std  = all_qpos_data.std(dim=0,  keepdim=True)
    qpos_std  = torch.clip(qpos_std, 0.1, np.inf)

    qvel_mean = all_qvel_data.mean(dim=0, keepdim=True)
    qvel_std  = all_qvel_data.std(dim=0,  keepdim=True)
    qvel_std  = torch.clip(qvel_std, 1e-2, np.inf)

    cf_mean = all_cf_data.mean(dim=0, keepdim=True)
    cf_std  = all_cf_data.std(dim=0,  keepdim=True)
    cf_std  = torch.clip(cf_std, 1e-2, np.inf)

    cf_zscored = (all_cf_data - cf_mean) / cf_std
    cf_z_max = cf_zscored.abs().max(dim=0).values.numpy().tolist()
    cf_z_99  = torch.quantile(cf_zscored.abs(), 0.99, dim=0).numpy().tolist()
    cf_z_95  = torch.quantile(cf_zscored.abs(), 0.95, dim=0).numpy().tolist()
    print(f'[get_norm_stats] cf z-score |max| per dim:    {[round(x,2) for x in cf_z_max]}')
    print(f'[get_norm_stats] cf z-score 99-percentile:    {[round(x,2) for x in cf_z_99]}')
    print(f'[get_norm_stats] cf z-score 95-percentile:    {[round(x,2) for x in cf_z_95]}')
    print(f'[get_norm_stats] applying clip = ±{cf_clip}')

    if cf_clip is not None and cf_clip > 0:
        clip_mask = (cf_zscored.abs() > cf_clip).any(dim=-1)
        clip_pct  = float(clip_mask.float().mean().item()) * 100
        print(f'[get_norm_stats] estimated clip-hit rate:     {clip_pct:.2f}% of frames')

    stats = {
        "action_mean":  action_mean.numpy().squeeze(),
        "action_std":   action_std.numpy().squeeze(),
        "qpos_mean":    qpos_mean.numpy().squeeze(),
        "qpos_std":     qpos_std.numpy().squeeze(),
        "qvel_mean":    qvel_mean.numpy().squeeze(),
        "qvel_std":     qvel_std.numpy().squeeze(),
        "cf_mean":      cf_mean.numpy().squeeze(),
        "cf_std":       cf_std.numpy().squeeze(),
        "cf_clip":      float(cf_clip) if cf_clip is not None else 0.0,
        "state_dim":    int(state_dim),
        "cf_dim":       int(cf_dim),
        "action_cf_dim": int(action_cf_dim),
        "force_substeps": int(force_substeps),
        "force_label_key": force_key_used or '/measured_cf',
        "example_qpos": all_qpos_data[0].numpy(),
    }
    return stats


def load_data(dataset_dir, num_episodes, camera_names,
              batch_size_train, batch_size_val,
              state_dim, chunk_size, cf_dim, action_cf_dim, cf_clip=3.0,
              force_substeps=0):
    f_total = action_cf_dim * max(int(force_substeps), 1)
    print(f'\nData from: {dataset_dir}')
    print(f'Using first {state_dim} dimensions of qpos/qvel/action')
    print(f'Using first {cf_dim} dimensions of measured_cf (obs)')
    if force_substeps > 0:
        print(f'方案2 substeps: K={force_substeps} → '
              f'Action = q_cmd({state_dim}) + f_cmd({force_substeps}×{action_cf_dim}='
              f'{f_total}) → action_dim {state_dim + f_total}')
    else:
        print(f'Action = q_cmd({state_dim}) + f_cmd({action_cf_dim}) '
              f'→ action_dim {state_dim + f_total}')
    print(f'CF z-score clip: ±{cf_clip}')
    print(f'Chunk size: {chunk_size}\n')

    train_ratio      = 0.8
    shuffled_indices = np.random.permutation(num_episodes)
    train_indices    = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices      = shuffled_indices[int(train_ratio * num_episodes):]

    norm_stats = get_norm_stats(dataset_dir, num_episodes, state_dim, cf_dim,
                                action_cf_dim, cf_clip=cf_clip,
                                force_substeps=force_substeps)

    train_dataset = EpisodicDataset(train_indices, dataset_dir, camera_names,
                                    norm_stats, state_dim, chunk_size, cf_dim,
                                    action_cf_dim, cf_clip=cf_clip,
                                    force_substeps=force_substeps)
    val_dataset   = EpisodicDataset(val_indices,   dataset_dir, camera_names,
                                    norm_stats, state_dim, chunk_size, cf_dim,
                                    action_cf_dim, cf_clip=cf_clip,
                                    force_substeps=force_substeps)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train,
                                  shuffle=True, pin_memory=True,
                                  num_workers=1, prefetch_factor=1)
    val_dataloader   = DataLoader(val_dataset,   batch_size=batch_size_val,
                                  shuffle=True, pin_memory=True,
                                  num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats, train_dataset.is_sim


### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])


def sample_insertion_pose():
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]
    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])
    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose


### helper functions

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result


def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
