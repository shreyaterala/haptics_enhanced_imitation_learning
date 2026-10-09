import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

from constants import DT, STATE_DIM, CF_DIM
from constants import PUPPET_GRIPPER_JOINT_OPEN
from utils import load_data
from utils import sample_box_pose, sample_insertion_pose
from utils import compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy, CNNMLPPolicy
from visualize_episodes import save_videos

from sim_env import BOX_POSE

import IPython
e = IPython.embed


# python imitate_episodes.py \
#   --task_name dvrk_retraction_u_nof \
#   --ckpt_dir ./checkpoints/dvrk_pickup_force6d_planB \
#   --policy_class ACT \
#   --batch_size 8 \
#   --seed 0 \
#   --num_epochs 5000 \
#   --lr 2e-6 \
#   --state_dim 7 \
#   --chunk_size 50 \
#   --hidden_dim 512 \
#   --dim_feedforward 3200 \
#   --kl_weight 10 \
#   --cf_clip 3.0 \
#   --temporal_agg


def main(args):
    set_seed(1)
    is_eval          = args['eval']
    ckpt_dir         = args['ckpt_dir']
    policy_class     = args['policy_class']
    onscreen_render  = args['onscreen_render']
    task_name        = args['task_name']
    batch_size_train = args['batch_size']
    batch_size_val   = args['batch_size']
    num_epochs       = args['num_epochs']

    # ------------------------------------------------------------------
    # state_dim: CLI --state_dim N specifies using first N dims; defaults to STATE_DIM
    # ------------------------------------------------------------------
    state_dim = args.get('state_dim') or STATE_DIM
    assert 1 <= state_dim <= STATE_DIM, \
        f'--state_dim must be between 1 and {STATE_DIM}, got: {state_dim}'

    # ── bilateral: action = [q_cmd | f_cmd] ────────────────────────────
    action_cf_dim = int(args.get('action_cf_dim')
                        if args.get('action_cf_dim') is not None else CF_DIM)
    # Scheme 2: when K > 0, f_cmd flattens K 1kHz substeps
    force_substeps = int(args.get('force_substeps') or 0)
    action_f_total = action_cf_dim * max(force_substeps, 1)
    action_dim = state_dim + action_f_total
    force_loss_weight = float(args.get('force_loss_weight')
                              if args.get('force_loss_weight') is not None else 1.0)

    print(f'[INFO] Training dimension: first {state_dim} dims (dataset raw dimension: {STATE_DIM})')
    print(f'[INFO] Network input tokens: qpos {state_dim} + qvel {state_dim} + force {CF_DIM} (projected independently)')
    if force_substeps > 0:
        print(f'[INFO] Scheme 2 substeps: K={force_substeps}')
        print(f'[INFO] Network output: action_dim {action_dim} = '
              f'q_cmd {state_dim} + f_cmd {force_substeps}×{action_cf_dim}')
    else:
        print(f'[INFO] Network output: action_dim {action_dim} = q_cmd {state_dim} + f_cmd {action_cf_dim}')
    print(f'[INFO] force_loss_weight: {force_loss_weight}')

    # cf_clip: z-score clip range. Saved in stats during training, loaded from stats during inference,
    # ensuring train/eval consistency
    cf_clip = float(args.get('cf_clip') if args.get('cf_clip') is not None else 3.0)
    print(f'[INFO] CF z-score clip: ±{cf_clip}')

    from constants import SIM_TASK_CONFIGS
    if task_name in SIM_TASK_CONFIGS:
        task_config = SIM_TASK_CONFIGS[task_name]
        is_sim = True
    else:
        from aloha_scripts.constants import TASK_CONFIGS
        task_config = TASK_CONFIGS[task_name]
        is_sim = False

    dataset_dir  = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len  = task_config['episode_len']
    camera_names = task_config['camera_names']

    lr_backbone = 1e-5
    backbone    = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads     = 8
        policy_config = {
            'lr':              args['lr'],
            'num_queries':     args['chunk_size'],
            'kl_weight':       args['kl_weight'],
            'hidden_dim':      args['hidden_dim'],
            'dim_feedforward': args['dim_feedforward'],
            'lr_backbone':     lr_backbone,
            'backbone':        backbone,
            'enc_layers':      enc_layers,
            'dec_layers':      dec_layers,
            'nheads':          nheads,
            'camera_names':    camera_names,
            'state_dim':       state_dim,
            'cf_dim':          CF_DIM,
            'action_cf_dim':   action_cf_dim,
            'force_substeps':  force_substeps,
            'action_dim':      action_dim,
            'force_loss_weight': force_loss_weight,
        }
    elif policy_class == 'CNNMLP':
        policy_config = {
            'lr':           args['lr'],
            'lr_backbone':  lr_backbone,
            'backbone':     backbone,
            'num_queries':  1,
            'camera_names': camera_names,
            'state_dim':    state_dim,
            'cf_dim':       CF_DIM,
        }
    else:
        raise NotImplementedError

    config = {
        'num_epochs':      num_epochs,
        'ckpt_dir':        ckpt_dir,
        'episode_len':     episode_len,
        'state_dim':       state_dim,
        'cf_dim':          CF_DIM,
        'action_cf_dim':   action_cf_dim,
        'force_substeps':  force_substeps,
        'action_dim':      action_dim,
        'cf_clip':         cf_clip,
        'chunk_size':      args['chunk_size'],
        'lr':              args['lr'],
        'policy_class':    policy_class,
        'onscreen_render': onscreen_render,
        'policy_config':   policy_config,
        'task_name':       task_name,
        'seed':            args['seed'],
        'temporal_agg':    args['temporal_agg'],
        'camera_names':    camera_names,
        'real_robot':      not is_sim,
    }

    if is_eval:
        ckpt_names = ['policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])
        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    # ----------------------------------------------------------------
    # Pass cf_clip to load_data; load_data will:
    #   1) Store it in norm_stats['cf_clip'] in get_norm_stats
    #   2) Apply it after cf z-score normalization in EpisodicDataset
    # ----------------------------------------------------------------
    train_dataloader, val_dataloader, norm_stats, _ = load_data(
        dataset_dir, num_episodes, camera_names,
        batch_size_train, batch_size_val,
        state_dim,
        args['chunk_size'],
        CF_DIM,
        action_cf_dim,
        cf_clip=cf_clip,
        force_substeps=force_substeps,
    )

    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Save stats (including cf_mean/std/clip) - run script reads all normalization params here
    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(norm_stats, f)
    print(f'[INFO] Saved stats: {stats_path}')
    print(f'  keys: {list(norm_stats.keys())}')

    config_path = os.path.join(ckpt_dir, 'train_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    ckpt_path = os.path.join(ckpt_dir, 'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir        = config['ckpt_dir']
    cf_dim          = config['cf_dim']
    state_dim       = config['state_dim']
    real_robot      = config['real_robot']
    policy_class    = config['policy_class']
    onscreen_render = config['onscreen_render']
    policy_config   = config['policy_config']
    camera_names    = config['camera_names']
    max_timesteps   = config['episode_len']
    task_name       = config['task_name']
    temporal_agg    = config['temporal_agg']
    onscreen_cam    = 'angle'

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')

    stats_path = os.path.join(ckpt_dir, 'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)

    # ── Read cf_clip from stats to ensure consistency with training ─────────
    cf_clip = float(stats.get('cf_clip', 0.0))
    if cf_clip > 0:
        print(f'[eval_bc] CF clip from stats: ±{cf_clip}')
    else:
        print('[eval_bc] No cf_clip in stats - skipping clip (legacy or unclipped training).')

    # ── pre_process / post_process ───────────────────────────────
    action_dim = int(config.get('action_dim', state_dim + config.get('action_cf_dim', cf_dim)))

    pre_process = lambda s_qpos: (s_qpos[:state_dim] - stats['qpos_mean']) / stats['qpos_std']
    pre_process_qvel = lambda s_qvel: (s_qvel[:state_dim] - stats['qvel_mean']) / stats['qvel_std']

    def pre_process_cf(s_cf):
        cf = (s_cf[:cf_dim] - stats['cf_mean']) / stats['cf_std']
        if cf_clip > 0:
            cf = np.clip(cf, -cf_clip, cf_clip)
        return cf

    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

    if real_robot:
        from aloha_scripts.robot_utils import move_grippers
        from aloha_scripts.real_env import make_real_env
        env = make_real_env(init_node=True)
        env_max_reward = 0
    else:
        from sim_env import make_sim_env
        env = make_sim_env(task_name)
        env_max_reward = env.task.max_reward

    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']

    max_timesteps = int(max_timesteps * 1)
    num_rollouts  = 50
    episode_returns  = []
    highest_rewards  = []

    for rollout_id in range(num_rollouts):
        if 'sim_transfer_cube' in task_name:
            BOX_POSE[0] = sample_box_pose()
        elif 'sim_insertion' in task_name:
            BOX_POSE[0] = np.concatenate(sample_insertion_pose())

        ts = env.reset()

        if onscreen_render:
            ax = plt.subplot()
            plt_img = ax.imshow(env._physics.render(height=480, width=640, camera_id=onscreen_cam))
            plt.ion()

        if temporal_agg:
            # bilateral: action = [q_cmd | f_cmd], buffer uses action_dim
            all_time_actions = torch.zeros(
                [max_timesteps, max_timesteps + num_queries, action_dim]).cuda()

        # ── Scheme B: save qpos and cf independently ─────────────────────────
        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        cf_history   = torch.zeros((1, max_timesteps, cf_dim)).cuda()
        image_list, qpos_list, target_qpos_list, rewards = [], [], [], []

        with torch.inference_mode():
            for t in range(max_timesteps):
                if onscreen_render:
                    image = env._physics.render(height=480, width=640, camera_id=onscreen_cam)
                    plt_img.set_data(image)
                    plt.pause(DT)

                obs = ts.observation
                image_list.append(obs['images'] if 'images' in obs else {'main': obs['image']})

                qpos_numpy = np.array(obs['qpos'])
                qvel_numpy = np.array(obs.get('qvel', np.zeros_like(qpos_numpy)))
                cf_numpy   = np.array(obs['cf'])    # ← adapt field name to your actual dvrk env name
                qpos_norm  = pre_process(qpos_numpy)
                qvel_norm  = pre_process_qvel(qvel_numpy)
                cf_norm    = pre_process_cf(cf_numpy)   # includes clipping

                # bilateral: qpos / qvel / cf passed independently into network
                qpos = torch.from_numpy(qpos_norm).float().cuda().unsqueeze(0)
                qvel = torch.from_numpy(qvel_norm).float().cuda().unsqueeze(0)
                cf   = torch.from_numpy(cf_norm).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                cf_history  [:, t] = cf
                curr_image = get_image(ts, camera_names)

                if config['policy_class'] == 'ACT':
                    if t % query_frequency == 0:
                        all_actions = policy(qpos, qvel, cf, curr_image)
                    if temporal_agg:
                        all_time_actions[[t], t:t + num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action  = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                elif config['policy_class'] == 'CNNMLP':
                    raw_action = policy(qpos, qvel, cf, curr_image)
                else:
                    raise NotImplementedError

                raw_action  = raw_action.squeeze(0).cpu().numpy()
                action      = post_process(raw_action)
                target_qpos = action[:state_dim]    # bilateral: first state_dim dims are position command
                target_cf   = action[state_dim:]    # remainder is f_cmd (typically unused in sim env)

                ts = env.step(target_qpos)

                qpos_list.append(qpos_numpy)
                target_qpos_list.append(target_qpos)
                rewards.append(ts.reward)

            plt.close()

        if real_robot:
            move_grippers([env.puppet_bot_left, env.puppet_bot_right],
                          [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards != None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, '
              f'{env_max_reward=}, Success: {episode_highest_reward == env_max_reward}')

        if save_episode:
            save_videos(image_list, DT,
                        video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
    avg_return   = np.mean(episode_returns)
    summary_str  = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env_max_reward + 1):
        more_or_equal_r      = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate * 100}%\n'

    print(summary_str)
    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    # bilateral: DataLoader returns 6-tuple
    image_data, qpos_data, qvel_data, cf_data, action_data, is_pad = data
    image_data, qpos_data, qvel_data, cf_data, action_data, is_pad = (
        image_data.cuda(), qpos_data.cuda(), qvel_data.cuda(), cf_data.cuda(),
        action_data.cuda(), is_pad.cuda())
    return policy(qpos_data, qvel_data, cf_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs    = config['num_epochs']
    ckpt_dir      = config['ckpt_dir']
    seed          = config['seed']
    policy_class  = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy    = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    # ── Sanity check: verify dimensions of first batch (Scheme B) ──────────
    try:
        first_batch = next(iter(train_dataloader))
        img_b, qpos_b, qvel_b, cf_b, act_b, pad_b = first_batch
        print(f'[sanity] batch shapes:')
        print(f'  image_data:  {tuple(img_b.shape)}')
        print(f'  qpos_data:   {tuple(qpos_b.shape)}  (expected: B, {config["state_dim"]})')
        print(f'  qvel_data:   {tuple(qvel_b.shape)}  (expected: B, {config["state_dim"]})')
        print(f'  cf_data:     {tuple(cf_b.shape)}    (expected: B, {config["cf_dim"]})')
        print(f'  action_data: {tuple(act_b.shape)}   (expected: B, chunk_size, {config["action_dim"]})')
        print(f'  is_pad:      {tuple(pad_b.shape)}')
        if qpos_b.shape[1] != config["state_dim"]:
            print(f'  ⚠️  qpos_data dimension incorrect! Should be {config["state_dim"]}')
        if cf_b.shape[1] != config["cf_dim"]:
            print(f'  ⚠️  cf_data   dimension incorrect! Should be {config["cf_dim"]}')
        if act_b.shape[2] != config["action_dim"]:
            print(f'  ⚠️  action_data dimension incorrect! Should be {config["action_dim"]} '
                  f'(= q_cmd {config["state_dim"]} + f_cmd {config["action_cf_dim"]})')
    except Exception as ex:
        print(f'[sanity] check failed: {ex}')

    train_history      = []
    validation_history = []
    min_val_loss   = np.inf
    best_ckpt_info = None

    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')

        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss   = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))

        epoch_summary = compute_dict_mean(
            train_history[(batch_idx + 1) * epoch:(batch_idx + 1) * (epoch + 1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # After epoch 0, print clip hit count if dataloader tracking is enabled
        if epoch == 0:
            try:
                ds = train_dataloader.dataset
                if hasattr(ds, 'total_count') and ds.total_count > 0:
                    pct = ds.clip_hit_count / ds.total_count * 100
                    print(f'[epoch 0] CF clip-hit rate: {pct:.2f}% '
                          f'({ds.clip_hit_count}/{ds.total_count} frames)')
            except Exception:
                pass

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values   = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)),      train_values, label='train')
        plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values,   label='validation')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval',            action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir',        action='store', type=str,   required=True)
    parser.add_argument('--policy_class',    action='store', type=str,   required=True)
    parser.add_argument('--task_name',       action='store', type=str,   required=True)
    parser.add_argument('--batch_size',      action='store', type=int,   required=True)
    parser.add_argument('--seed',            action='store', type=int,   required=True)
    parser.add_argument('--num_epochs',      action='store', type=int,   required=True)
    parser.add_argument('--lr',              action='store', type=float, required=True)

    parser.add_argument('--state_dim', action='store', type=int, default=None,
                        help=f'Number of state dimensions used for training, range 1 ~ {STATE_DIM}, default = {STATE_DIM}')

    # bilateral: action force target
    parser.add_argument('--action_cf_dim', action='store', type=int, default=None,
                        help='Dimension of f_cmd in action (default = CF_DIM).'
                             'action_dim = state_dim + action_cf_dim')
    parser.add_argument('--force_loss_weight', action='store', type=float, default=1.0,
                        help='Weight for f_cmd portion of L1 loss')
    parser.add_argument('--force_substeps', action='store', type=int, default=0,
                        help='Scheme 2: flatten K high-frequency force substeps per action step '
                             '(K = round(1000/policy_fps), requires running '
                             'tools/build_force_substeps.py first). 0 = disabled (Scheme 1)')

    # CF (force sensor) settings
    parser.add_argument('--cf_clip', action='store', type=float, default=3.0,
                        help='CF z-score clip range (±cf_clip). Prevents force sensor outliers'
                             'from corrupting force tokens. Default 3.0, typical range 2.0 ~ 5.0.'
                             'Set to 0 to disable clipping.')

    # for ACT
    parser.add_argument('--kl_weight',       action='store', type=int,  required=False)
    parser.add_argument('--chunk_size',      action='store', type=int,  required=False)
    parser.add_argument('--hidden_dim',      action='store', type=int,  required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int,  required=False)
    parser.add_argument('--temporal_agg',    action='store_true')

    main(vars(parser.parse_args()))