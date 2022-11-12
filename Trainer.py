import numpy as np
from HeartsGym import Hearts
from Agent import Agent, RandomAgent
from Memory import Memory, HeartsMemoryLoader
import torch

class Trainer:
    
    PATH = "Agents/checkpoint"
    
    def __init__(self,logging_iteration, max_iteration=-1):
        
        self.logging_iteration = logging_iteration
        
        if max_iteration == -1:
            self.max_iteration = 9999999999999999
        else:
            self.max_iteration = max_iteration
            
    def train(self,agentType):

        environment = Hearts()
        agent = agentType(environment.observation_space.n,environment.action_space.n)
        # if we have a saved model, continue training
        try:
            checkpoint = torch.load(self.PATH)
            iteration = checkpoint["iteration"] + 1
            agent.model.load_state_dict(checkpoint["model"])
            agent.optimizer.load_state_dict(checkpoint["optimizer"])

            print(f"Resuming training from iteration {iteration}")
        except:
            iteration = 1
            
            
        memory = Memory(max_size=10000)
        loader = HeartsMemoryLoader()

        losses = 0 
        averageReward = 0
        lossesList = []
        averageRewardList = []   
        
        for iteration in range(iteration, self.max_iteration + 1):
            done = False
            state = environment.reset()

            while not done:
                action, q = agent.act(state)
                currentPlayer = environment.getCurrentPlayer()

                next_state, reward, done = environment.step(action)

                if loader.hasStartState(currentPlayer):
                    loader.loadNextState(state,currentPlayer)
                    memory.push(loader.ejectElement(currentPlayer))

                if not loader.hasStartState(currentPlayer):
                    loader.loadStartState(state,currentPlayer)
                    loader.loadAction(action, currentPlayer)
                
                if environment.isRoundOver():
                    for i in range(4):
                        loader.loadReward(reward[i],i)
                    
                state = next_state
                
            avgReward =  self.test(agent,RandomAgent(),RandomAgent(),RandomAgent(),1)
            averageReward += avgReward[0]
            

            for i in range(4):
                loader.loadNextState(state,i)
                memory.push(loader.ejectElement(i))

            for _ in range(32):
                memory_batch = memory.get_batch(batch_size=64)
                loss = agent.update(memory_batch)
            
            
            
            losses += loss
            lossesList.append(loss)
            agent.update_randomness()

            if iteration % self.logging_iteration == 0:
                print(f"Iteration: {iteration}")
                print(f"  Loss: {(losses/self.logging_iteration):.5f}")
                print(f"  Average Reward: {(averageReward/self.logging_iteration):.5f}")
                print()
                
                checkpoint = {
                    "iteration": iteration,
                    "model": agent.model.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                }
                
                torch.save(checkpoint, self.PATH)

                averageRewardList.append(averageReward/self.logging_iteration)
                
                losses = 0
                averageReward = 0
                
        return lossesList, averageRewardList
    
    def test(self,agent1,agent2,agent3,agent4,testIteration):
        
        environment = Hearts()
        
        agents = [agent1,
                  agent2,
                  agent3,
                  agent4]
        
        totalRewards = [0,0,0,0]
        
        for iteration in range(1, testIteration + 1):
            done = False
            state = environment.reset()

            while not done:
                currentPlayer = environment.getCurrentPlayer()
                action, q = agents[currentPlayer].act(state)
                
                next_state, reward, done = environment.step(action)

                state = next_state
            
                
            totalRewards[0] += reward[0]
            totalRewards[1] += reward[1]
            totalRewards[2] += reward[2]
            totalRewards[3] += reward[3]
            
        totalRewards[0] /= testIteration
        totalRewards[1] /= testIteration
        totalRewards[2] /= testIteration
        totalRewards[3] /= testIteration
        
        return totalRewards
        
                
            