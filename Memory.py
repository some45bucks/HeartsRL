from asyncio.windows_events import NULL
import random
from collections import deque

class HeartsMemoryLoader():
    def __init__(self):
        self.startStates = [[],[],[],[]]
        self.actions = [-1,-1,-1,-1]
        self.nextStates = [[],[],[],[]]
        self.rewards = [-1,-1,-1,-1]

    def loadStartState(self,state,spot):
        self.startStates[spot] = state

    def hasStartState(self,spot):
        return len(self.startStates[spot]) != 0

    def loadNextState(self,state,spot):
        self.nextStates[spot] = state
   
    def loadReward(self,reward,spot):
        self.rewards[spot] = reward

    def loadAction(self,action,spot):
        self.actions[spot] = action

    def ejectElement(self,spot):
        element = [self.startStates[spot],self.actions[spot],self.nextStates[spot],self.rewards[spot]]

        self.startStates[spot] = []
        self.actions[spot] = -1
        self.nextStates[spot] = []
        self.rewards[spot] = -1

        return element

class Memory(object):
    def __init__(self, max_size=100):
        self.memory = deque(maxlen=max_size)

    def push(self, element):
        self.memory.append(element)

    def get_batch(self, batch_size=4):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        return random.sample(self.memory, batch_size)

    def __repr__(self):
        return f"Current elements in memory: {len(self.memory)}"

    def __len__(self):
        return len(self.memory)