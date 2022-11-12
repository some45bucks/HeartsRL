#%%
import matplotlib.pyplot as plt

from Trainer import Trainer
from Agent import Agent

trainer = Trainer(500)

losses, averageRewardList = trainer.train(Agent)


#%%
plt.plot(losses)
plt.title('Loss over time')
plt.xlabel('Loss')
plt.ylabel('Episode')
plt.show()

plt.plot(averageRewardList)
plt.title('Reward over time')
plt.xlabel('Reward')
plt.ylabel('Episode')
plt.show()
# %%
