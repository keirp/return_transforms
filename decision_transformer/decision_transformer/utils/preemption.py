import torch
import os
import pickle
import wandb
import time

class CheckpointTimer:

	def __init__(self, checkpoint_every):
		self.checkpoint_every = checkpoint_every
		self.last_chk = 0

	def should(self):
		now = time.time()
		return now - self.last_chk >= self.checkpoint_every

	def done(self):
		now = time.time()
		self.last_chk = now

class PreemptionManager:

	def __init__(self, checkpoint_dir, checkpoint_every=0, checkpoint_timer=None, prefix=''):
		self.checkpoint_dir = checkpoint_dir
		self._wandb_id = None
		self.prefix = prefix
		if checkpoint_timer is None:
			self.checkpoint_timer = CheckpointTimer(checkpoint_every)
		else:
			self.checkpoint_timer = checkpoint_timer
		self.last_chk = 0
		self.stored = dict()

	def _load_data(self, name):
		if self.checkpoint_dir is not None:
			path = os.path.join(self.checkpoint_dir, f'{self.prefix}_{name}.pkl')
			if os.path.exists(path):
				with open(path, 'rb') as file:
					print(f'Loaded {name}...')
					data = pickle.load(file)
				return data
		return None

	def save(self, name, data, now=False):
		if now:
			if self.checkpoint_dir is not None:
				path = os.path.join(self.checkpoint_dir, f'{self.prefix}_{name}.pkl')
				with open(path, 'wb') as path:
					pickle.dump(data, path)
		else:
			self.stored[name] = data

	def wandb_id(self):
		if self._wandb_id is not None:
			return self._wandb_id

		self._wandb_id = self._load_data('wandb_id')

		if self._wandb_id is None:
			self._wandb_id = wandb.util.generate_id()

			self.save('wandb_id', self._wandb_id, now=True)

		return self._wandb_id

	def load_torch(self, name, cl, *args, **kwargs):
		state_dict = self._load_data(name)
		model = cl(*args, **kwargs)
		if state_dict is not None:
			model.load_state_dict(state_dict)

		return model

	def exists(self, name):
		if self.checkpoint_dir is not None:
			path = os.path.join(self.checkpoint_dir, f'{self.prefix}_{name}.pkl')
			return os.path.exists(path)
		return False

	def save_torch(self, name, model):
		self.save(name, model.state_dict())

	def load_if_exists(self, name, value):
		data = self._load_data(name)
		if data is None:
			return value
		return data

	def for_obj(self, prefix):
		return PreemptionManager(self.checkpoint_dir, prefix=prefix, checkpoint_timer=self.checkpoint_timer)

	def checkpoint(self):
		if self.checkpoint_timer.should():
			for key in self.stored:
				self.save(key, self.stored[key], now=True)

			self.checkpoint_timer.done()
			self.stored = dict()
