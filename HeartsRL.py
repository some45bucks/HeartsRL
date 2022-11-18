#%%
import matplotlib.pyplot as plt

from Trainer import Trainer
from Agent import Agent

trainer = Trainer(500,200000)

losses, averageRewardList = trainer.train(Agent)


#%%
import torch

PATH = "Agents/checkpoint"
checkpoint = torch.load(PATH)

lossesList = checkpoint["losses"]
averageRewardList = checkpoint["rewards"]

plt.plot(lossesList)
plt.title('Loss over time')
plt.ylabel('Loss')
plt.xlabel('Episode')
plt.show()

plt.plot(averageRewardList)
plt.title('Reward over time')
plt.ylabel('Reward')
plt.xlabel('Episode')
plt.show()
# %%
