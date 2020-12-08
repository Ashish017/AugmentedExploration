import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os

dirs = os.listdir()

seeds = [1,2,3]
ex_alphas_off = [0.1,0.2,0.4,0.8]
ex_alphas_sac = [0.1]
alphas = [0.1,0.2,0.4,0.8]
algos = ["SAC","Off_policy"]

'''
plot_num = 0
i = 0
for alpha in alphas:
	for algo in algos:
		if algo == "SAC":
			ex_aps = ex_alphas_sac
		else:
			ex_aps = ex_alphas_off

		for ex_alpha in ex_aps:
			plot_num+=1
			plt.subplot(len(alphas),len(ex_alphas_off)+1, plot_num)

			ep_lens = []
			ep_success = 0
			arrays = []
			for seed in seeds:
				name = algo + "_{}_{}_{}_all_states.npy".format(seed, alpha, ex_alpha)
				
				try:
					array = np.load(name)
					ep_success+=1
				except:
					continue

				x = array[:,0]
				y = array[:,1]

				ep_lens.append(len(x))

				plt_alpha = 300/len(x)
				plt.axis("off")
				plt.scatter(x,y,s=0.5,alpha=plt_alpha,color='yellow')

			avg_ep_len = round(sum(ep_lens)/ep_success)
			success_rate = round(ep_success*100/len(seeds))
			plt.title("Avg episode len: {}\nEpisode succes rate: {}%".format(avg_ep_len,success_rate), fontsize=6)
			plt.xlabel("das")
			plt.gca()

plt.tight_layout()
plt.show()
'''


plot_num = 0
i = 0
fig, axes = plt.subplots(len(alphas),len(ex_alphas_off)+1, figsize=(10,8))

for i,alpha in enumerate(alphas):
	for algo in algos:
		if algo == "SAC":
			ex_aps = ex_alphas_sac
		else:
			ex_aps = ex_alphas_off

		for j,ex_alpha in enumerate(ex_aps):
			plt_pos = [i,j+1]
			if algo == "SAC":
				plt_pos = [i,0]
			ax = axes[plt_pos[0], plt_pos[1]]
			ep_lens = []
			ep_success = 0
			arrays = []
			for seed in seeds:
				name = algo + "_{}_{}_{}_all_states.npy".format(seed, alpha, ex_alpha)
				try:
					array = np.load(name)
					ep_success+=1
				except:
					continue

				x = array[:,0]
				y = array[:,1]

				ep_lens.append(len(x))

				plt_alpha = 300/len(x)
				ax.axis("off")
				ax.scatter(x,y,s=0.5,alpha=plt_alpha,color='slateblue')
				ax.set_facecolor("black")

			avg_ep_len = round(sum(ep_lens)/ep_success)
			success_rate = round(ep_success*100/len(seeds))
			ax.set_title("Avg episode steps: {}\nEpisode succes rate: {}%".format(avg_ep_len,success_rate), fontsize=10)
			
plt.tight_layout()
#plt.show()

fig.savefig('whatever.png', facecolor='white', edgecolor='none')