import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms

from detr.main import build_ACT_model_and_optimizer, build_CNNMLP_model_and_optimizer
import IPython
e = IPython.embed


class ACTPolicy(nn.Module):
    """
    Bilateral ACT:
      输入: qpos, qvel, cf, image
      输出: action chunk, 每步 = [q_cmd (state_dim) | f_cmd (action_cf_dim)]
      损失: L = L1(q_cmd) + force_loss_weight * L1(f_cmd) + kl_weight * KL
    """
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_ACT_model_and_optimizer(args_override)
        self.model = model  # CVAE decoder
        self.optimizer = optimizer
        self.kl_weight = args_override['kl_weight']
        self.state_dim = args_override['state_dim']
        self.force_loss_weight = args_override.get('force_loss_weight', 1.0)
        print(f'KL Weight {self.kl_weight}, Force loss weight {self.force_loss_weight}')

    def __call__(self, qpos, qvel, cf, image, actions=None, is_pad=None):
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:  # training time
            actions = actions[:, :self.model.num_queries]
            is_pad  = is_pad[:, :self.model.num_queries]

            a_hat, is_pad_hat, (mu, logvar) = self.model(qpos, qvel, cf, image,
                                                         env_state, actions, is_pad)
            total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

            all_l1 = F.l1_loss(actions, a_hat, reduction='none')   # (bs, seq, action_dim)
            all_l1 = all_l1 * ~is_pad.unsqueeze(-1)

            sd = self.state_dim
            l1_pos   = all_l1[:, :, :sd].mean()
            l1_force = all_l1[:, :, sd:].mean() if a_hat.shape[-1] > sd \
                       else all_l1.new_zeros(())

            loss_dict = dict()
            loss_dict['l1']       = l1_pos
            loss_dict['l1_force'] = l1_force
            loss_dict['kl']       = total_kld[0]
            loss_dict['loss'] = (loss_dict['l1']
                                 + self.force_loss_weight * loss_dict['l1_force']
                                 + loss_dict['kl'] * self.kl_weight)
            return loss_dict
        else:                    # inference time
            a_hat, _, (_, _) = self.model(qpos, qvel, cf, image, env_state)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


class CNNMLPPolicy(nn.Module):
    def __init__(self, args_override):
        super().__init__()
        model, optimizer = build_CNNMLP_model_and_optimizer(args_override)
        self.model = model
        self.optimizer = optimizer

    def __call__(self, qpos, qvel, cf, image, actions=None, is_pad=None):
        # 注意: CNNMLP 这里没接 qvel/force,参数留作签名兼容
        env_state = None
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        image = normalize(image)
        if actions is not None:
            actions = actions[:, 0]
            a_hat = self.model(qpos, image, env_state, actions)
            mse = F.mse_loss(actions, a_hat)
            loss_dict = dict()
            loss_dict['mse']  = mse
            loss_dict['loss'] = loss_dict['mse']
            return loss_dict
        else:
            a_hat = self.model(qpos, image, env_state)
            return a_hat

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld          = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld           = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
