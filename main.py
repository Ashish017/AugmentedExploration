import gym, torch, random
import numpy as np
from parameters import settings
from trainer import SAC, Off_policy


if __name__ == '__main__':

	#Settings seeds
	torch.manual_seed(settings.seed)
	np.random.seed(settings.seed)
	random.seed(settings.seed)
	
	#Choosing algorithm
	if settings.algo == "SAC":
		trainer = SAC()
	elif settings.algo == "Off_policy":
		trainer = Off_policy()

	#Starting training
	trainer.start()