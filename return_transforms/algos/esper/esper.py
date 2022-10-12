from return_transforms.models.esper.cluster_model import ClusterModel
from return_transforms.models.esper.dynamics_model import DynamicsModel
from return_transforms.datasets.esper_dataset import ESPERDataset
from return_transforms.utils.utils import learned_labels
from tqdm.autonotebook import tqdm
import torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import gym


def esper(trajs,
          action_space,
          dynamics_model_args,
          cluster_model_args,
          train_args,
          device,
          n_cpu=2):

    # Check if discrete action space
    if isinstance(action_space, gym.spaces.Discrete):
        action_size = action_space.n
        act_loss_fn = lambda pred, truth: F.cross_entropy(pred.view(-1, pred.shape[-1]), torch.argmax(truth, dim=-1).view(-1),
                                                          reduction='none')
        act_type = 'discrete'
    else:
        action_size = action_space.shape[0]
        act_loss_fn = lambda pred, truth: ((pred - truth) ** 2).mean(dim=-1)
        act_type = 'continuous'

    # Get the length of the longest trajectory
    max_len = max([len(traj.obs) for traj in trajs]) + 1

    dataset = ESPERDataset(trajs, action_size, max_len,
                           gamma=train_args['gamma'], act_type=act_type)

    # Get the obs size from the first datapoint
    obs, _, _, _ = next(iter(dataset))
    obs_shape = obs[0].shape
    obs_size = np.prod(obs_shape)

    # Set up the models
    dynamics_model = DynamicsModel(obs_size,
                                   action_size,
                                   cluster_model_args['rep_size'],
                                   dynamics_model_args).to(device)

    cluster_model = ClusterModel(obs_size,
                                 action_size,
                                 cluster_model_args['rep_size'],
                                 cluster_model_args,
                                 cluster_model_args['groups']).to(device)

    dynamics_optimizer = optim.AdamW(
        dynamics_model.parameters(), lr=train_args['dynamics_model_lr'])
    cluster_optimizer = optim.AdamW(
        cluster_model.parameters(), lr=train_args['cluster_model_lr'])

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=train_args['batch_size'],
                                             num_workers=n_cpu)

    # Calculate epoch markers
    total_epochs = train_args['cluster_epochs'] + train_args['return_epochs']
    ret_stage = train_args['cluster_epochs']

    dynamics_model.train()
    cluster_model.train()
    for epoch in range(total_epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        total_loss = 0
        total_act_loss = 0
        total_ret_loss = 0
        total_dyn_loss = 0
        total_baseline_dyn_loss = 0
        for obs, acts, ret, seq_len in pbar:
            # Take an optimization step for the cluster model
            cluster_optimizer.zero_grad()
            obs = obs.to(device)
            acts = acts.to(device)
            ret = ret.to(device)
            seq_len = seq_len.to(device)

            mask = (acts.sum(dim=-1) == 0)

            # Get the cluster predictions
            clusters, ret_pred, act_pred, _ = cluster_model(
                obs, acts, ret, seq_len, hard=epoch >= ret_stage)

            pred_next_obs, next_obs = dynamics_model(
                obs, acts, clusters, seq_len)

            # Calculate the losses
            ret_loss = ((ret_pred - ret) ** 2)[~mask].mean()
            act_loss = act_loss_fn(act_pred, acts)[~mask].mean()
            dynamics_loss = ((pred_next_obs - next_obs) ** 2).mean()

            # Calculate the total loss
            if epoch < ret_stage:
                loss = train_args['adv_loss_weight'] * dynamics_loss + \
                    train_args['act_loss_weight'] * act_loss
            else:
                loss = ret_loss

            loss.backward()
            cluster_optimizer.step()

            # Take an optimization step for the dynamics model
            dynamics_optimizer.zero_grad()
            pred_next_obs, next_obs = dynamics_model(
                obs, acts, clusters.detach(), seq_len)
            baseline_pred_next_obs, _ = dynamics_model(
                obs, acts, torch.zeros_like(clusters), seq_len)
            dynamics_loss = ((pred_next_obs - next_obs) ** 2).mean()
            baseline_dynamics_loss = (
                (baseline_pred_next_obs - next_obs) ** 2).mean()
            total_dynamics_loss = dynamics_loss + baseline_dynamics_loss
            total_dynamics_loss.backward()
            dynamics_optimizer.step()

            # Update the progress bar
            total_loss += loss.item()
            total_act_loss += act_loss.item()
            total_ret_loss += ret_loss.item()
            total_dyn_loss += dynamics_loss.item()
            total_baseline_dyn_loss += baseline_dynamics_loss.item()

            advantage = baseline_dynamics_loss.item() - dynamics_loss.item()

            pbar.set_description(
                f"Epoch {epoch} | Loss: {total_loss / (len(pbar) + 1):.4f} | Act Loss: {total_act_loss / (len(pbar) + 1):.4f} | Rep Loss: {total_ret_loss / (len(pbar) + 1):.4f} | Dyn Loss: {total_dyn_loss / (len(pbar) + 1):.4f} | Adv: {advantage:.4f}")

    # Get the learned return labels
    avg_returns = []
    for traj in trajs:
        labels = learned_labels(traj, cluster_model,
                                action_size, max_len, device, act_type)
        avg_returns.append(labels)

    return avg_returns
