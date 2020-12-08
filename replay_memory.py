import random
import numpy as np
from parameters import hyperparameters as p
from parameters import settings

class Replay_buffer:

	def __init__(self, capacity):
		self.capacity = capacity
		self.buffer = []
		self.position = 0

	def push(self, data):
		if len(self.buffer) < self.capacity:
			self.buffer.append(None)
		self.buffer[self.position] = data
		self.position = int((self.position + 1) % self.capacity)

	def sample(self, batch_size):
		batch = random.sample(self.buffer, batch_size)
		return map(np.stack, zip(*batch))

	def save_all_states(self):
		all_states = []
		for i in range(len(self.buffer)):
			all_states.append(self.buffer[i][0])

		all_states = np.array(all_states)

		np.save("Data/{}_{}_{}_{}_all_states".format(settings.algo, settings.seed, round(p.alpha.item(),1), p.ex_alpha), all_states)

	def __len__(self):
		return len(self.buffer)

class Normalizer:

	def __init__(self, size):
		self.size = size
		
		self.total_sum = np.zeros(self.size, np.float32)
		self.total_sum_sq = np.zeros(self.size, np.float32)
		self.total_count = np.ones(1, np.float32)

		self.mean = np.zeros(self.size, np.float32)
		self.std = np.ones(self.size, np.float32)

	def update(self, v):
		try:
			self.total_count += 1
			self.total_sum += v
			self.total_sum_sq += np.square(v)

			self.mean = self.total_sum / self.total_count
			self.std = np.sqrt(np.maximum(np.square(p.eps), (self.total_sum_sq / self.total_count) - np.square(self.total_sum / self.total_count)))

		except:
			self.total_count += v.shape[0] - 1 	# -1 for +1 caused in try
			self.total_sum += v.sum()
			self.total_sum_sq += np.square(v).sum()

			self.mean = self.total_sum / self.total_count
			self.std = np.sqrt(np.maximum(np.square(p.eps), (self.total_sum_sq / self.total_count) - np.square(self.total_sum / self.total_count)))

	def normalize(self, v):
		prev_shape = v.shape
		v = v.reshape(-1, self.size)
		v = np.clip((v - self.mean) / (self.std), -p.norm_clip_range, p.norm_clip_range)
		v = v.reshape(*prev_shape)
		return v