#%%
from Trainer import Trainer
from Agent import Agent

trainer = Trainer(500,400000)

losses, averageRewardList = trainer.trainMonte(Agent)


#%%
import torch
import matplotlib.pyplot as plt

PATH = "Agents/TDAgents/checkpoint400000half"
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
