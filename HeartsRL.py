#%%
import matplotlib.pyplot as plt

from Trainer import Trainer
from Agent import Agent

trainer = Trainer(500,400000)

losses, averageRewardList = trainer.train(Agent)


#%%
plt.plot(losses)
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
