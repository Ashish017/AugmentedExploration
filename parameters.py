class Hyperparameters:

	def __init__(self):
		#Network parameters
		self.hidden_dims = 256		# Dimensions of hidden layers
		self.log_std_min = -20.0	# Min value for log_std (used in Gaussian policy)
		self.log_std_max = 2.0		# Max value for log_std (used in Gaussian policy)

		self.tau = 0.005
		self.lr = 3e-4
		self.alpha = 0.2
		self.ex_alpha = 0.0
		self.gamma = 0.99
		self.lam = 0.95
		self.clip_range = 0.2

		self.max_numsteps = 2*1e6
		self.random_steps = 0
		
		self.exploitory_batch_size = 256
		self.exploitory_policy_memory_size = 1e6
		self.exploitory_policy_updates_per_steps = 1
		self.exploitory_target_update_interval = 1
		
		self.exploratory_batch_size = 256
		self.exploratory_policy_memory_size = 1e6
		self.exploratory_policy_updates_per_steps = 1
		self.exploratory_target_update_interval = 1

		self.ex_reward_model_updates_per_steps = 1

		self.reward_model_batch_size = 256

		self.testing_episodes = 5
		self.test_freq = 10

		self.norm_clip_range = 10
		self.eps = 0.01

class Settings:

	def __init__(self):
		self.algo = "Off_policy"
		self.reward_model = 'novelty'	#More reward models to be added later
		self.env_name = "Ant-v2"
		self.seed = 1

		self.device = 'cuda'		#cpu or cuda (if available)
		self.automatic_entropy_tuning = False
		self.automatic_ex_entropy_tuning = False	# Incomplete right now, keep False	

settings = Settings()
hyperparameters = Hyperparameters()