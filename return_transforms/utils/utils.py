import torch
import numpy as np

"""ESPER utils."""


def get_past_indices(x, seq_len):
    """
    Note: this assumes that padding is actually before the sequence.

    Often we want to get a tensor of indices for another tensor of shape
    (bsz, T, ...). These indices (bsz, T) should be between the start of the
    non-padded inputs and T. This function returns such an index tensor.
    """
    bsz, t = x.shape[:2]

    idxs = torch.randint(0, t, (bsz, t)).to(x)
    ts = torch.arange(0, t).view(1, t).expand(bsz, t).to(x)
    # Denotes how much padding is before each sequence
    pad_lens = t - seq_len.view(bsz, 1).expand(bsz, t)
    ts = ts - pad_lens + 1  # Shifts the indices so that the first non-padded index is 0

    # If ts == 0, then set idxs to 0. Otherwise, use the remainder of the division.
    idxs = torch.where(ts == 0, torch.zeros_like(idxs), idxs % ts)

    # Now add back the padding lengths
    idxs = idxs + pad_lens

    return idxs.long()


def return_labels(traj, gamma=1):
    rewards = traj.rewards
    returns = []
    ret = 0
    for reward in reversed(rewards):
        ret *= gamma
        ret += float(reward)
        returns.append(ret)
    returns = list(reversed(returns))
    return returns


def learned_labels(traj, label_model, n_actions, horizon, device,
                   act_type='discrete'):
    with torch.no_grad():
        label_model.eval()
        obs = np.array(traj.obs)
        if act_type == 'discrete':
            a = np.array(traj.actions)
            actions = np.zeros((a.size, n_actions))
            actions[np.arange(a.size), a] = 1
        else:
            actions = np.array(traj.actions)

        labels = []

        padded_obs = np.zeros((horizon, *obs.shape[1:]))
        padded_acts = np.zeros((horizon, n_actions))

        padded_obs[-obs.shape[0]:] = obs
        padded_acts[-obs.shape[0]:] = actions

        padded_obs = torch.tensor(padded_obs).float().unsqueeze(0).to(device)
        padded_acts = torch.tensor(padded_acts).float().unsqueeze(0).to(device)

        labels, _ = label_model.return_preds(
            padded_obs, padded_acts, hard=True)
        labels = labels[0, -obs.shape[0]:].view(-1).cpu().detach().numpy()

    return np.around(labels, decimals=1)
