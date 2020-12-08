import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from parameters import hyperparameters as p

class ValueNet(nn.Module):
	def __init__(self, state_dims):
		super(ValueNet, self).__init__()

		self.layers = nn.Sequential(
										nn.Linear(state_dims, p.hidden_dims),
										nn.ReLU(),
										nn.Linear(p.hidden_dims, p.hidden_dims),
										nn.ReLU(),
										nn.Linear(p.hidden_dims, 1)
									)

		init_weights(self.layers)

	def forward(self, state):
		return self.layers(state)

class QNet(nn.Module):
	def __init__(self, state_dims, action_dims):
		super(QNet, self).__init__()

		self.layers1 = nn.Sequential(
										nn.Linear(state_dims + action_dims, p.hidden_dims),
										nn.ReLU(),
										nn.Linear(p.hidden_dims, p.hidden_dims),
										nn.ReLU(),
										nn.Linear(p.hidden_dims, 1)
									)
		self.layers2 = nn.Sequential(
										nn.Linear(state_dims + action_dims, p.hidden_dims),
										nn.ReLU(),
										nn.Linear(p.hidden_dims, p.hidden_dims),
										nn.ReLU(),
										nn.Linear(p.hidden_dims, 1)
									)

		init_weights(self.layers1)
		init_weights(self.layers2)

	def forward(self, state, action):
		x = torch.cat((state, action),1)
		return self.layers1(x), self.layers2(x)

class GaussianPolicy(nn.Module):
	def __init__(self, state_dims, action_dims):
		super(GaussianPolicy, self).__init__()

		self.layers = nn.Sequential(
										nn.Linear(state_dims, p.hidden_dims),
										nn.ReLU(),
										nn.Linear(p.hidden_dims, p.hidden_dims),
										nn.ReLU(),
									)

		self.action_layer = nn.Linear(p.hidden_dims, action_dims)
		self.log_std_layer = nn.Linear(p.hidden_dims, action_dims)

		init_weights(self.layers)

	def forward(self, state):
		x = self.layers(state)
		mean_action = self.action_layer(x)
		log_std = self.log_std_layer(x)
		log_std = torch.clamp(log_std, min=p.log_std_min, max=p.log_std_max)
		return mean_action, log_std

	def sample(self, state):
		mean, log_std = self.forward(state)

		std = log_std.exp()
		normal = Normal(mean, std)
		
		x_t = normal.rsample()
		y_t = torch.tanh(x_t)
		action = y_t
		log_prob = normal.log_prob(x_t)
		log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
		log_prob = log_prob.sum(1, keepdim=True)
		mean = torch.tanh(mean)
		return action, log_prob, mean

	def get_logprob(self, state, action):
		mean, log_std = self.forward(state)

		std = log_std.exp()
		normal = Normal(mean, std)

		x_t = torch.atanh(action)
		y_t = torch.tanh(x_t)
		log_prob = normal.log_prob(x_t)
		log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)

		return log_prob

class FCN(nn.Module):

	def __init__(self, input_size, output_size):
		super(FCN, self).__init__()

		self.layers = nn.Sequential(
									nn.Linear(input_size, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, p.hidden_dims),
									nn.ReLU(),
									nn.Linear(p.hidden_dims, output_size),
									)
		init_weights(self.layers)
		
	def forward(self, state):
		return self.layers(state)
		
def init_weights(layers):
	for layer in layers:
		if hasattr(layer, 'weight') or hasattr(layer, "bias"):
			for name, param in layer.named_parameters():
					if name == "weight":
						nn.init.kaiming_uniform_(param, nonlinearity='leaky_relu')
					if name == "bias":
						nn.init.constant_(param, 0.0)