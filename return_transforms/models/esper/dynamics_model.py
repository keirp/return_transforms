import torch
import torch.nn as nn
from return_transforms.models.basic.mlp import MLP
from return_transforms.utils.utils import get_past_indices


class DynamicsModel(nn.Module):
    """
    Dynamics predictor that conditions on a trajectory representation. 
    Uses MLPs, but can easily be extended to different architectures.

    During training, this model conditions itself on a trajectory representation
    from a random timestep in the past.
    """

    def __init__(self, obs_size, action_size, rep_size, model_args):
        super().__init__()
        self.obs_size = obs_size
        self.action_size = action_size
        self.rep_size = rep_size
        self.dynamics_model = MLP(
            obs_size + action_size + rep_size, obs_size, **model_args)

    def forward(self, obs, action, cluster, seq_len):
        bsz, t = obs.shape[:2]
        obs = obs.view(bsz, t, -1)

        x = torch.cat([obs, action], dim=-1)

        idxs = get_past_indices(x, seq_len)
        idxs = idxs.view(bsz, t, 1).expand(bsz, t, self.rep_size)

        past_cluster = torch.gather(cluster, 1, idxs)

        # We don't condition on the last timestep since we don't have a next observation
        context = x[:, :-1]
        context = torch.cat([context, past_cluster[:, :-1]], dim=-1)
        next_obs = obs[:, 1:]

        pred_next_obs = self.dynamics_model(
            context.view(bsz * (t - 1), -1)).view(*next_obs.shape)

        return pred_next_obs, next_obs
