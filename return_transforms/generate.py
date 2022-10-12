from email import generator
from return_transforms.algos.esper.esper import esper
from fire import Fire
import yaml
from pathlib import Path


def load_config(config_path):
    return yaml.safe_load(Path(config_path).read_text())


def load_env(env_name):
    if env_name == 'connect_4':
        from esper_envs.envs.offline_envs.connect_four_offline_env import ConnectFourOfflineEnv
        from esper_envs.envs.connect_four.connect_four_env import GridWrapper
        # TODO: env should just deal with this automatically
        c4_ds_path = '../esper_envs/offline_data/c4data_mdp.ds'
        c4_exec_dir = '../esper_envs/connect4'
        task = ConnectFourOfflineEnv(c4_ds_path, exec_dir=c4_exec_dir)
        env = task.env_cls()
        env = GridWrapper(env)
        return env, task.trajs
    # TODO: implement the rest


def generate(env_name, config, device, n_cpu=2):
    config = load_config(config)

    if config['method'] == 'esper':
        env, trajs = load_env(env_name)
        rets = esper(trajs,
                     env.action_space,
                     config['dynamics_model_args'],
                     config['cluster_model_args'],
                     config['train_args'],
                     device,
                     n_cpu)

    else:
        raise NotImplementedError


if __name__ == '__main__':
    Fire(generate)
