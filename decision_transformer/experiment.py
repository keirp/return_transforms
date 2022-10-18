import gym
import numpy as np
import torch
import wandb
from torch import nn

import argparse
import pickle
import random
import sys

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer

import d4rl
from decision_transformer.utils.preemption import PreemptionManager
from stoch_rvs.utils.utils import return_labels, learned_labels, set_seed
from stoch_rvs.algos.learn_labels import learn_labels

from stoch_rvs.datasets.seq_dataset import SeqDataset
from decision_transformer.utils.convert_dataset import convert_dataset


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def experiment(
        exp_prefix,
        variant,
):
    pm = PreemptionManager(variant['checkpoint_dir'], checkpoint_every=600)

    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, dataset = variant['env'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    ret_postprocess_fn = lambda returns: returns
    action_type = 'continuous'
    if env_name == 'hopper':
        env = gym.make('Hopper-v3')
        max_ep_len = 1000
        env_targets = [3600, 1800]  # evaluation conditioning targets
        scale = 1000.  # normalization for rewards/returns
        name = f'{env_name}-{dataset}-v2'
        d4rl_env = gym.make(name)
        ret_postprocess_fn = d4rl_env.get_normalized_score
    elif env_name == 'halfcheetah':
        env = gym.make('HalfCheetah-v3')
        max_ep_len = 1000
        env_targets = [12000, 6000]
        scale = 1000.
        name = f'{env_name}-{dataset}-v2'
        d4rl_env = gym.make(name)
        ret_postprocess_fn = d4rl_env.get_normalized_score
    elif env_name == 'walker2d':
        env = gym.make('Walker2d-v3')
        max_ep_len = 1000
        env_targets = [5000, 2500]
        scale = 1000.
        name = f'{env_name}-{dataset}-v2'
        d4rl_env = gym.make(name)
        ret_postprocess_fn = d4rl_env.get_normalized_score
    elif env_name == 'reacher2d':
        from decision_transformer.envs.reacher_2d import Reacher2dEnv
        env = Reacher2dEnv()
        max_ep_len = 100
        env_targets = [76, 40]
        scale = 10.
    elif env_name == 'gambling':
        from stochastic_offline_envs.envs.offline_envs.gambling_offline_env import GamblingOfflineEnv
        task = GamblingOfflineEnv()
        env = task.env_cls()
        max_ep_len = 5
        env_targets = list(np.arange(-15, 5, 0.5)) + [5]
        # env_targets = [-15, -6, 1, 5]
        scale = 5.
        action_type = 'discrete'
    elif env_name == 'connect_four':
        from stochastic_offline_envs.envs.offline_envs.connect_four_offline_env import ConnectFourOfflineEnv
        from stochastic_offline_envs.envs.connect_four.connect_four_env import GridWrapper
        # TODO: env should just deal with this automatically
        task = ConnectFourOfflineEnv()
        env_cls = lambda: GridWrapper(task.env_cls())
        env = env_cls()
        max_ep_len = 50
        env_targets = list(np.arange(-1, 1, 0.25)) + [1]
        # env_targets = [-1, 0, 1]
        scale = 1.
        action_type = 'discrete'
    elif env_name == 'tfe':
        from stochastic_offline_envs.envs.offline_envs.tfe_offline_env import TFEOfflineEnv
        task = TFEOfflineEnv()
        env = task.env_cls()
        max_ep_len = 500
        env_targets = list(np.arange(0, 1, 0.1)) + [1]
        scale = 1.
        action_type = 'discrete'
    else:
        raise NotImplementedError

    if model_type == 'bc':
        # since BC ignores target, no need for different evaluations
        env_targets = env_targets[:1]

    example_state = env.reset()
    state_dim = np.prod(env.observation_space.shape)

    if action_type == 'discrete':
        act_dim = env.action_space.n
    else:
        act_dim = env.action_space.shape[0]

    # load dataset
    dataset_path = f'data/{env_name}-{dataset}-v2.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    n_data = len(trajectories)
    used_data = int(n_data * variant['prop_data'])
    trajectories = trajectories[:used_data]

    esper_trajs = convert_dataset(trajectories, action_type)

    # save all path information into separate lists
    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        if mode == 'delayed':  # delayed: all rewards moved to end of trajectory
            path['rewards'][-1] = path['rewards'].sum()
            path['rewards'][:-1] = 0.
        # Pre-compute the return-to-gos
        path['rtg'] = discount_cumsum(path['rewards'], gamma=1.)
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    traj_lens, returns = np.array(traj_lens), np.array(returns)

    # used for input normalization
    states = np.concatenate(states, axis=0)
    state_mean, state_std = np.mean(
        states, axis=0), np.std(states, axis=0) + 1e-6

    num_timesteps = sum(traj_lens)

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {dataset}')
    print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
    print(
        f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
    print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
    print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns)  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[sorted_inds[-1]]
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] <= num_timesteps:
        timesteps += traj_lens[sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[sorted_inds] / sum(traj_lens[sorted_inds])

    def get_batch(batch_size=256, max_len=K):
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations']
                     [si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >=
                          max_ep_len] = max_ep_len - 1  # padding cutoff
            rtg.append(traj['rtg'][si:][:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1],
                                         np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len -
                                   tlen, state_dim)), s[-1]], axis=1)
            if variant['normalize_states']:
                s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len -
                                   tlen, act_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len -
                                   tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen))
                                   * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)),
                                     rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate(
                [np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate(
                [np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(
            dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(
            dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(
            dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(
            dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(
            dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(
            dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths = [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():
                    if model_type == 'dt':
                        ret, length = evaluate_episode_rtg(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            scale=scale,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                            action_type=action_type
                        )
                    else:
                        ret, length = evaluate_episode(
                            env,
                            state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret_postprocess_fn(ret))
                lengths.append(length)
            return {
                f'target_{target_rew}_return_mean': np.mean(returns),
                # f'target_{target_rew}_return_std': np.std(returns),
                # f'target_{target_rew}_length_mean': np.mean(lengths),
                # f'target_{target_rew}_length_std': np.std(lengths),
            }
        return fn

    if model_type == 'dt':
        model = pm.load_torch('model', DecisionTransformer,
                              state_dim=state_dim,
                              act_dim=act_dim,
                              max_length=K,
                              max_ep_len=max_ep_len,
                              hidden_size=variant['embed_dim'],
                              n_layer=variant['n_layer'],
                              n_head=variant['n_head'],
                              n_inner=4 * variant['embed_dim'],
                              activation_function=variant['activation_function'],
                              n_positions=1024,
                              resid_pdrop=variant['dropout'],
                              attn_pdrop=variant['dropout'],
                              action_tanh=action_type == 'continuous',
                              rtg_seq=variant['rtg_seq'])
    elif model_type == 'bc':
        model = pm.load_torch('model', MLPBCModel,
                              state_dim=state_dim,
                              act_dim=act_dim,
                              max_length=K,
                              hidden_size=variant['embed_dim'],
                              n_layer=variant['n_layer'],)
    else:
        raise NotImplementedError

    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']
    optimizer = pm.load_torch('optimizer', torch.optim.AdamW,
                              model.parameters(),
                              lr=variant['learning_rate'],
                              weight_decay=variant['weight_decay'])
    scheduler = pm.load_torch('scheduler', torch.optim.lr_scheduler.LambdaLR,
                              optimizer,
                              lambda steps: min((steps + 1) / warmup_steps, 1))

    if action_type == 'continuous':
        action_loss = lambda s_hat, a_hat, r_hat, s, a, r: torch.mean(
            (a_hat - a)**2)
    else:
        ce_loss = nn.CrossEntropyLoss()

        def action_loss(s_hat, a_hat, r_hat, s, a, r):
            a = torch.argmax(a, dim=-1)
            return ce_loss(a_hat, a)

    if model_type == 'dt':
        trainer = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=action_loss,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=action_loss,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb_id = pm.wandb_id()
        wandb.init(
            name=exp_prefix,
            # group=group_name,
            project='decision-transformer',
            config=variant,
            id=wandb_id,
            resume='allow',
        )
        # wandb.watch(model)  # wandb has some bug

    if variant['rtg']:
        # Load custom return-to-go
        rtg_path = variant['rtg']
        # Load the pickle
        with open(rtg_path, 'rb') as f:
            rtg_dict = pickle.load(f)
        for i, path in enumerate(trajectories):
            path['rtg'] = rtg_dict[i]

    # if mode == 'esper':
    #     if variant['normalize_states']:
    #         print('Normalizing states')
    #         for traj in esper_trajs:
    #             for i in range(len(traj.obs)):
    #                 traj.obs[i] = (traj.obs[i] - state_mean) / state_std
    #     print(esper_trajs[0].obs[0])
    #     seq_dataset = SeqDataset(esper_trajs, act_dim, max_ep_len, gamma=1, reward_norm=scale,
    #                              act_type=action_type)
    #     label_model = learn_labels(seq_dataset,
    #                                act_dim,
    #                                batch_size=100,
    #                                learning_rate=5e-4,
    #                                hidden_size=512,
    #                                rep_size=variant['rep_size'],
    #                                rep_groups=variant['rep_groups'],
    #                                device='cuda',
    #                                pm=pm.for_obj('label_model'),
    #                                act_loss_coef=variant['act_loss_coef'],
    #                                adv_loss_coef=variant['adv_loss_coef'],
    #                                pretrain_epochs=0,
    #                                cluster_epochs=variant['cluster_epochs'],
    #                                label_epochs=variant['label_epochs'])
    #     label_fn = lambda traj: learned_labels(
    #         traj, label_model, act_dim, max_ep_len, 'cuda', act_type=action_type)
    #     labs = [label_fn(traj) * scale for traj in esper_trajs]
    #     for i, path in enumerate(trajectories):
    #         path['rtg'] = labs[i]

    completed_iters = pm.load_if_exists('completed_iters', 0)
    for iter in range(completed_iters, variant['max_iters']):
        outputs = trainer.train_iteration(
            num_steps=variant['num_steps_per_iter'], iter_num=iter + 1, print_logs=True)
        pm.save_torch('optimizer', optimizer)
        pm.save_torch('scheduler', scheduler)
        pm.save_torch('model', model)
        pm.checkpoint()
        if log_to_wandb:
            wandb.log(outputs)
        completed_iters += 1
        pm.save('completed_iters', completed_iters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='hopper')
    # medium, medium-replay, medium-expert, expert
    parser.add_argument('--dataset', type=str, default='medium')
    # normal for standard setting, delayed for sparse, esper for learned average returns
    parser.add_argument('--mode', type=str, default='normal')
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--model_type', type=str, default='dt')
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=10)
    parser.add_argument('--num_steps_per_iter', type=int, default=10000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)
    parser.add_argument('--checkpoint_dir', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--act_loss_coef', type=float, default=0.01)
    parser.add_argument('--adv_loss_coef', type=float, default=1)
    parser.add_argument('--cluster_epochs', type=int, default=5)
    parser.add_argument('--label_epochs', type=int, default=5)
    parser.add_argument('--rep_size', type=int, default=8)
    parser.add_argument('--rep_groups', type=int, default=1)
    parser.add_argument('--rtg_seq', type=bool, default=True)
    parser.add_argument('--normalize_states', action='store_true')

    parser.add_argument('--prop_data', type=float, default=1.)

    parser.add_argument('--rtg', type=str, default=None)

    args = parser.parse_args()

    print(vars(args))

    experiment('gym-experiment', variant=vars(args))
