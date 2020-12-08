import torch, gym, itertools, time, pybulletgym, pybullet_envs
from torch.optim import Adam
import torch.nn.functional as F
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from model import ValueNet, QNet, GaussianPolicy
from replay_memory import Replay_buffer, Normalizer
from parameters import settings
from parameters import hyperparameters as p
from exploration_rewards import Novelty

class Algo:

	def __init__(self):
		#Creating environment
		self.env = gym.make(settings.env_name)
		self.env.seed(settings.seed)
		self.env.action_space.seed(settings.seed)

		self.state_space = self.env.observation_space.shape[0]
		self.action_space = self.env.action_space.shape[0]

		self.obs_normalizer = Normalizer(self.state_space)

		self.device = torch.device(settings.device)
		self.writer = SummaryWriter('runs/' + settings.env_name + "_" + settings.algo + '_{}_{}_{}'.format(p.alpha, p.ex_alpha, settings.seed))

		#Initializing common networks and their optimizers
		self.exploitory_policy = GaussianPolicy(self.state_space, self.action_space).to(self.device)
		self.exploitory_Q = QNet(self.state_space, self.action_space).to(self.device)
		self.exploitory_Q_target = QNet(self.state_space, self.action_space).to(self.device)
		self.exploitory_policy_optim = Adam(self.exploitory_policy.parameters(), lr=p.lr)
		self.exploitory_Q_optim = Adam(self.exploitory_Q.parameters(), lr=p.lr)
	
		self.target_update(self.exploitory_Q_target, self.exploitory_Q, 1.0)

		p.alpha = torch.Tensor([p.alpha]).to(self.device)
		if settings.automatic_entropy_tuning:
			self.target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
			self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
			self.alpha_optim = Adam([self.log_alpha], lr = p.lr)

		if settings.automatic_ex_entropy_tuning:
			self.ex_target_entropy = -torch.prod(torch.Tensor(self.env.action_space.shape).to(self.device)).item()
			self.ex_log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
			self.ex_alpha_optim = Adam([self.log_alpha], lr = p.lr)			

		if settings.reward_model == 'novelty':
			self.ex_reward_model = Novelty(self.state_space, self.device)
		
	def target_update(self, target, source, tau=p.tau):
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

	def update_exploitory_policy(self, memory):
		state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(p.exploitory_batch_size)
		state_batch, next_state_batch = self.obs_normalizer.normalize(state_batch), self.obs_normalizer.normalize(next_state_batch)

		state_batch = torch.FloatTensor(state_batch).to(self.device)
		action_batch = torch.FloatTensor(action_batch).to(self.device)
		reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
		next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
		mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

		with torch.no_grad():
			next_state_action, next_state_log_pi, _ = self.exploitory_policy.sample(next_state_batch)
			qf1_next_target, qf2_next_target = self.exploitory_Q_target(next_state_batch, next_state_action)
			min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - p.alpha * next_state_log_pi
			next_q_value = reward_batch + mask_batch * p.gamma * (min_qf_next_target)

		qf1, qf2 = self.exploitory_Q(state_batch, action_batch)

		qf1_loss = F.mse_loss(qf1, next_q_value)
		qf2_loss = F.mse_loss(qf2, next_q_value)
		qf_loss = qf1_loss + qf2_loss

		self.exploitory_Q_optim.zero_grad()
		qf_loss.backward()
		self.exploitory_Q_optim.step()

		pi, log_pi, _ = self.exploitory_policy.sample(state_batch)
		qf1_pi, qf2_pi = self.exploitory_Q(state_batch, pi)
		min_qf_pi = torch.min(qf1_pi, qf2_pi)
		policy_loss = ((p.alpha * log_pi) - min_qf_pi).mean()

		self.exploitory_policy_optim.zero_grad()
		policy_loss.backward()
		self.exploitory_policy_optim.step()

		alpha_loss = torch.Tensor([0.0])

		if settings.automatic_entropy_tuning:
			alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
			self.alpha_optim.zero_grad()
			alpha_loss.backward()
			self.alpha_optim.step()

			p.alpha = self.log_alpha.exp().item()

		ex_reward_model_loss = self.ex_reward_model.update(memory)

		return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), alpha_loss.item(), p.alpha, ex_reward_model_loss

	def test_current_policy(self):
		avg_reward = 0
		avg_steps = 0
		avg_ex_rewards = 0

		for episode in range(p.testing_episodes):
			episode_steps = 0
			state = self.env.reset()
			episode_rewards = 0
			episode_ex_rewards = 0
			done = False

			while not done:
				episode_steps += 1
				norm_state = self.obs_normalizer.normalize(state)
				action = self.select_action(norm_state, self.exploitory_policy, evaluate=True)
				next_state, reward, done, _ = self.env.step(action)
				episode_rewards += reward

				state = next_state
			
			avg_reward += episode_rewards
			avg_ex_rewards += episode_ex_rewards
			avg_steps += episode_steps

		avg_reward = avg_reward/p.testing_episodes
		avg_ex_rewards = avg_ex_rewards/p.testing_episodes
		avg_steps = avg_steps/p.testing_episodes

		return avg_reward, avg_steps

	def select_action(self, state, policy, evaluate=False):
		with torch.no_grad():
			try:
				state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
				if evaluate is False:
					action, log_prob, _ = policy.sample(state)
				else:
					_, log_prob, action = policy.sample(state)

				return action.cpu().numpy()[0]

			except:
				state = state.unsqueeze(0)
				if evaluate is False:
					action, log_prob, _ = policy.sample(state)
				else:
					_, log_prob, action = policy.sample(state)

				return action

	def log(self, data):
		for key in data.keys():
			if key != "total_numsteps":
				self.writer.add_scalar(key.split('_')[-1]+"/"+key, data[key], data['total_numsteps'])
		print("Total number of Steps: {} \t Average reward per episode: {}".format(data['total_numsteps'], round(data['average_rewards'],1)))

	def start(self):
		raise NotImplementedError

class SAC(Algo):

	def __init__(self):
		super(SAC, self).__init__()
		self.memory = Replay_buffer(capacity = p.exploitory_policy_memory_size)
		
	def start(self):
		total_numsteps = 0

		for episode in itertools.count(1):
			episode_rewards = 0.0
			episode_steps = 0
			episode_ex_rewards = 0.0
			done = False
			state = self.env.reset()

			while not done:
				episode_steps += 1
				if p.random_steps > total_numsteps:
					action = self.env.action_space.sample()
				else:
					norm_state = self.obs_normalizer.normalize(state)
					action = self.select_action(norm_state, self.exploitory_policy)

				if len(self.memory) > p.exploitory_batch_size:
					for i in range(p.exploitory_policy_updates_per_steps):
						qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha, ex_reward_model_loss = self.update_exploitory_policy(self.memory)
						if episode % p.exploitory_target_update_interval == 0:
							self.target_update(self.exploitory_Q_target, self.exploitory_Q, p.tau)

				next_state, reward, done, _ = self.env.step(action)

				tensor_state = torch.FloatTensor(state).to(self.device)
				tensor_next_state = torch.FloatTensor(next_state).to(self.device)
				episode_ex_rewards += self.ex_reward_model.get_reward(tensor_state, tensor_next_state).item()

				total_numsteps += 1
				episode_rewards += reward

				# Ignore the done signal if it comes from hitting the time horizon.
				mask = 1.0 if episode_steps == self.env._max_episode_steps else float(not done)

				self.memory.push((state, action, reward, next_state, mask))
				self.obs_normalizer.update(state)
				state = next_state

			if episode % p.test_freq == 0:
				average_rewards, average_episode_steps = self.test_current_policy()
				try:
					data = {
								'average_rewards'			: average_rewards,
								'total_numsteps'			: total_numsteps,
								'average_episode_steps'		: average_episode_steps,
								'qf1_loss'					: qf1_loss,
								'qf2_loss'					: qf2_loss,
								'exploitory_policy_loss'	: policy_loss,
								'alpha_loss'				: alpha_loss,
								'alpha_value'				: alpha,
								'per_step_avg_ex_rewards'	: episode_ex_rewards / episode_steps,
								'ex_reward_model_loss'		: ex_reward_model_loss

					}
					self.log(data)
				except UnboundLocalError:
					pass
				
			if total_numsteps > p.max_numsteps:
				self.env.close()
				self.writer.close()
				break

class Off_policy(Algo):

	def __init__(self):
		super(Off_policy, self).__init__()
		self.memory = Replay_buffer(capacity = p.exploitory_policy_memory_size)
		self.exploratory_policy = GaussianPolicy(self.state_space, self.action_space).to(self.device)
		self.exploratory_Q = QNet(self.state_space, self.action_space).to(self.device)
		self.exploratory_Q_target = QNet(self.state_space, self.action_space).to(self.device)
		self.exploratory_policy_optim = Adam(self.exploratory_policy.parameters(), lr=p.lr)
		self.exploratory_Q_optim = Adam(self.exploratory_Q.parameters(), lr=p.lr)

		self.target_update(self.exploratory_policy, self.exploitory_policy, 1.0)

		self.kl_normalizer = Normalizer(1)
		self.ex_rewards_normalizer = Normalizer(1)

	def start(self):
		total_numsteps = 0

		for episode in itertools.count(1):
			episode_rewards = 0.0
			episode_steps = 0
			done = False
			state = self.env.reset()

			while not done:
				episode_steps += 1
				if p.random_steps > total_numsteps:
					action = self.env.action_space.sample()
				else:
					norm_state = self.obs_normalizer.normalize(state)
					action = self.select_action(norm_state, self.exploratory_policy)

				if len(self.memory) > p.exploitory_batch_size and len(self.memory) > p.exploratory_batch_size:
					for i in range(p.exploitory_policy_updates_per_steps):
						qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha, ex_reward_model_loss = self.update_exploitory_policy(self.memory)
						if episode % p.exploitory_target_update_interval == 0:
							self.target_update(self.exploitory_Q_target, self.exploitory_Q, p.tau)

					for i in range(p.exploratory_policy_updates_per_steps):
						ex_qf1_loss, ex_qf2_loss, ex_policy_loss, divergence_loss = self.update_exploratory_policy(self.memory)
						if episode % p.exploratory_target_update_interval == 0:
							self.target_update(self.exploratory_Q_target, self.exploratory_Q, p.tau)
		
				next_state, reward, done, _ = self.env.step(action)
				total_numsteps += 1
				episode_rewards += reward

				# Ignore the done signal if it comes from hitting the time horizon.
				mask = 1.0 if episode_steps == self.env._max_episode_steps else float(not done)

				self.memory.push((state, action, reward, next_state, mask))
				self.obs_normalizer.update(state)
				state = next_state

			if episode % p.test_freq == 0:
				average_rewards, average_episode_steps = self.test_current_policy()
				try:
					
					data = {
								'average_rewards'			: average_rewards,
								'total_numsteps'			: total_numsteps,
								'average_episode_steps'		: average_episode_steps,
								'qf1_loss'					: qf1_loss,
								'qf2_loss'					: qf2_loss,
								'exploitory_policy_loss'	: policy_loss,
								'alpha_loss'				: alpha_loss,
								'alpha_value'				: alpha,
								'ex_qf1_loss'				: ex_qf1_loss,
								'ex_qf2_loss'				: ex_qf2_loss,
								'ex_policy_loss'			: ex_policy_loss,
								'ex_reward_model_loss'		: ex_reward_model_loss,
								'divergence_loss'			: divergence_loss
					}
					
					self.log(data)
				except UnboundLocalError:
					pass

			if total_numsteps > p.max_numsteps:
				self.env.close()
				self.writer.close()
				break

	def update_exploratory_policy(self, memory):
		state_batch, action_batch, reward_batch, next_state_batch, mask_batch = memory.sample(p.exploitory_batch_size)
		state_batch, next_state_batch = self.obs_normalizer.normalize(state_batch), self.obs_normalizer.normalize(next_state_batch)
		
		state_batch = torch.FloatTensor(state_batch).to(self.device)
		action_batch = torch.FloatTensor(action_batch).to(self.device)
		reward_batch = torch.FloatTensor(reward_batch).to(self.device).unsqueeze(1)
		next_state_batch = torch.FloatTensor(next_state_batch).to(self.device)
		mask_batch = torch.FloatTensor(mask_batch).to(self.device).unsqueeze(1)

		with torch.no_grad():
			ex_rewards = self.ex_reward_model.get_reward(state_batch, next_state_batch)
			ex_rewards = ex_rewards.unsqueeze(1).cpu().numpy()
			ex_reward_batch = self.ex_rewards_normalizer.normalize(ex_rewards)
			self.ex_rewards_normalizer.update(ex_rewards)
			ex_reward_batch = torch.FloatTensor(ex_reward_batch).to(self.device)

			ex_next_state_action, ex_next_state_log_pi, _ = self.exploratory_policy.sample(next_state_batch)
			qf1_next_target, qf2_next_target = self.exploratory_Q_target(next_state_batch, ex_next_state_action)
			
			'''
			ex_mean_actions, ex_log_std = self.exploratory_policy(next_state_batch)
			mean_actions, log_std = self.exploitory_policy(next_state_batch)
			ex_normal = Normal(ex_mean_actions, ex_log_std.exp())
			normal = Normal(mean_actions, log_std.exp())
			kl_div = torch.distributions.kl_divergence(ex_normal, normal).mean(1).unsqueeze(1)
			'''
			
			ex_next_state_log_prob = torch.clamp(self.exploratory_policy.get_logprob(next_state_batch, ex_next_state_action), min=p.log_std_min, max=p.log_std_max)
			next_state_log_prob = torch.clamp(self.exploitory_policy.get_logprob(next_state_batch, ex_next_state_action), min=p.log_std_min, max=p.log_std_max)
			
			kl_div = (ex_next_state_log_prob - next_state_log_prob).mean(1).unsqueeze(1)

			min_qf_next_target = p.ex_alpha*(torch.min(qf1_next_target,qf2_next_target)-(p.alpha*ex_next_state_log_pi)) - kl_div
			next_q_value = ex_reward_batch + mask_batch * p.gamma * (min_qf_next_target)

		qf1, qf2 = self.exploratory_Q(state_batch, action_batch)

		qf1_loss = F.mse_loss(qf1, next_q_value)
		qf2_loss = F.mse_loss(qf2, next_q_value)
		qf_loss = qf1_loss + qf2_loss

		self.exploratory_Q_optim.zero_grad()
		qf_loss.backward()
		self.exploratory_Q_optim.step()

		ex_pi, ex_log_pi, _ = self.exploratory_policy.sample(state_batch)
		
		qf1_pi, qf2_pi = self.exploratory_Q(state_batch, ex_pi)
		min_qf_pi = torch.min(qf1_pi, qf2_pi)

		'''
		ex_mean_actions, ex_log_std = self.exploratory_policy(state_batch)
		mean_actions, log_std = self.exploitory_policy(state_batch)
		ex_normal = Normal(ex_mean_actions, ex_log_std.exp())
		normal = Normal(mean_actions, log_std.exp())
		kl_div = torch.distributions.kl_divergence(ex_normal, normal).mean(1).unsqueeze(1)
		'''

		ex_state_log_prob = torch.clamp(self.exploratory_policy.get_logprob(state_batch, ex_pi), min=p.log_std_min, max=p.log_std_max)
		with torch.no_grad(): state_log_prob = torch.clamp(self.exploitory_policy.get_logprob(state_batch, ex_pi), min=p.log_std_min, max=p.log_std_max)
		kl_div = (ex_state_log_prob - state_log_prob).mean(1).unsqueeze(1)

		policy_loss = (p.ex_alpha*((p.alpha * ex_log_pi) - min_qf_pi) + kl_div).mean()

		self.exploratory_policy_optim.zero_grad()
		policy_loss.backward()
		self.exploratory_policy_optim.step()

		ex_alpha_loss = torch.Tensor([0.0])

		if settings.automatic_ex_entropy_tuning:
			ex_alpha_loss = -(self.ex_log_alpha * (ex_log_pi + self.ex_target_entropy).detach()).mean()
			self.ex_alpha_optim.zero_grad()
			ex_alpha_loss.backward()
			self.ex_alpha_optim.step()

			p.ex_alpha = self.ex_log_alpha.exp().item()

		return qf1_loss.item(), qf2_loss.item(), policy_loss.item(), kl_div.mean().item()