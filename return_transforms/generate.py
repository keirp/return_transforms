from email import generator
import pickle
from return_transforms.algos.esper.esper import esper
from fire import Fire
import yaml
from pathlib import Path


def load_config(config_path):
    return yaml.safe_load(Path(config_path).read_text())


def load_env(env_name):
    if env_name == 'connect_four':
        from esper_envs.envs.offline_envs.connect_four_offline_env import ConnectFourOfflineEnv
        from esper_envs.envs.connect_four.connect_four_env import GridWrapper
        # TODO: env should just deal with this automatically
        c4_ds_path = '../esper_envs/offline_data/c4data_mdp.ds'
        c4_exec_dir = '../esper_envs/connect4'
        task = ConnectFourOfflineEnv(c4_ds_path, exec_dir=c4_exec_dir)
        env = task.env_cls()
        env = GridWrapper(env)
        trajs = task.trajs
        for traj in trajs:
            for i in range(len(traj.obs)):
                traj.obs[i] = traj.obs[i]['grid']
        return env, trajs
    elif env_name == 'tfe':
        from esper_envs.envs.offline_envs.tfe_offline_env import TFEOfflineEnv
        tfe_ds_path = '../esper_envs/offline_data/2048_5m_4x4.ds'
        task = TFEOfflineEnv(path=tfe_ds_path)
        env = task.env_cls()
        trajs = task.trajs
        return env, trajs
    elif env_name == 'gambling':
        from esper_envs.envs.offline_envs.gambling_offline_env import GamblingOfflineEnv
        gambling_ds_path = '../esper_envs/offline_data/gambling.ds'
        task = GamblingOfflineEnv(path=gambling_ds_path)
        env = task.env_cls()
        trajs = task.trajs
        return env, trajs
    # TODO: implement the rest


def generate(env_name, config, ret_file, device, n_cpu=2):
    print('Loading config...')
    config = load_config(config)

    if config['method'] == 'esper':
        print('Loading offline RL task...')
        env, trajs = load_env(env_name)
        print('Creating ESPER returns...')
        rets = esper(trajs,
                     env.action_space,
                     config['dynamics_model_args'],
                     config['cluster_model_args'],
                     config['train_args'],
                     device,
                     n_cpu)

        # Save the returns as a pickle
        print('Saving returns...')
        Path(ret_file).parent.mkdir(parents=True, exist_ok=True)
        with open(ret_file, 'wb') as f:
            pickle.dump(rets, f)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    Fire(generate)
