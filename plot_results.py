import matplotlib.pyplot as plt
import numpy as np

rewards = np.load("/Users/stephanehatgiskessell/Downloads/episode_rewards.npy")
x = np.linspace(0,len(rewards)-1,len(rewards))

plt.plot(x,rewards)
plt.show()
