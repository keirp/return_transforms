from stoch_rvs.samplers.trajectory_sampler import Trajectory
import numpy as np
from copy import deepcopy

def convert_dataset(trajectories, action_type):
	trajs = []
	for path in trajectories:
		obs_ = []
		actions_ = []
		rewards_ = []
		infos_ = []
		policy_infos_ = []
		for t in range(len(path['observations'])):
			obs_.append(deepcopy(path['observations'][t]))
			if action_type == 'discrete':
				actions_.append(np.argmax(path['actions'][t]))
			else:
				actions_.append(path['actions'][t])
			rewards_.append(path['rewards'][t])
		trajs.append(Trajectory(obs=obs_,
		                        actions=actions_, 
		                        rewards=rewards_,
		                        infos=infos_, 
		                        policy_infos=policy_infos_))
	return trajs