import torch
from model import FCN
import torch.nn.functional as F
from torch.optim import Adam
from parameters import hyperparameters as p

class Novelty:

	def __init__(self, state_dims, device):
		self.device = device
		self.model = FCN(state_dims, state_dims).to(device)
		self.model_optim = Adam(self.model.parameters(), lr=p.lr)

	def get_reward(self, state, next_state):
		with torch.no_grad():
			pred_next_state = self.model(state)
			try:
				reward = ((pred_next_state - next_state)**2).mean(1)
			except IndexError:
				reward = ((pred_next_state - next_state)**2).mean()
		return reward

	def update(self, memory):
		state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(p.reward_model_batch_size)
		state_batch = torch.FloatTensor(state_batch).to(self.device)
		next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
		pred_next_states = self.model(state_batch)
		loss = F.mse_loss(next_state_batch, pred_next_states)

		self.model_optim.zero_grad()
		loss.backward()
		self.model_optim.step()

		return loss.item()