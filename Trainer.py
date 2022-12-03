import numpy as np
from HeartsGym import Hearts
from Agent import Agent, RandomAgent
from Memory import Memory, HeartsMemoryLoader
import torch
import copy

class TreeNode:
    def __init__(self,state,returnState,reward=[],gameDone=False):
        self.state = copy.deepcopy(state)
        self.returnState = copy.deepcopy(returnState)
        self.children = [0,0,0,0,0,0,0,0,0,0,0,0,0]
        self.gameDone = gameDone
        self.reward = reward

    def expand(self,agent,env,random=.001):
        
        if self.gameDone:
            return 
        
        _, probs = agent.act(self.returnState)
        
        for i in range(len(probs)):
            env.state = copy.deepcopy(self.state)
            if (probs[i] > 0 or random < np.random.rand()) and env.isActionValid(i):
                next_state, reward, done = env.step(i)
                
                self.children[i] = TreeNode(env.state,next_state,reward,done)
                
    def drill(self,agent,env):
        
        if self.gameDone:
            return
        
        _, probs = agent.act(self.returnState)
        
        best = torch.argmax(probs).item()
        
        env.state = copy.deepcopy(self.state)
        
        next_state, reward, done = env.step(best)
                
        self.children[best] = TreeNode(env.state,next_state,reward,done)
        
        self.children[best].drill(agent,env)
        
    def calcExpectedReward(self):
        
        meanReward = np.zeros(4)
        
        count = 0
        
        for child in self.children:
            if child != 0:
                count += 1
                
                childReward = child.calcExpectedReward()
                
                meanReward[0] += childReward[0]
                meanReward[1] += childReward[1]
                meanReward[2] += childReward[2]
                meanReward[3] += childReward[3]
        
        if count != 0:
            meanReward /= count
            self.reward = meanReward
        else:
            return self.reward
        
        return meanReward
            
            
                
        
        

class HeartGameTree:
    
    def __init__(self,numberOfGames):
        self.numberOfGames = numberOfGames
    
    def search(self,rState,environment,agent,random=.001):
        
        root = TreeNode(environment.state,rState)
        
        possibleExpansions = [root]
        
        while len(possibleExpansions) < self.numberOfGames and len(possibleExpansions) > 0:

            random_index = np.random.randint(0, len(possibleExpansions))
            node = possibleExpansions.pop(random_index) 
                  
            node.expand(agent,environment,random)
            
            for child in node.children:
                if child != 0 and not child.gameDone and len(possibleExpansions) < self.numberOfGames:
                    possibleExpansions.append(child)
        
        for leafNode in possibleExpansions:
            leafNode.drill(agent,environment)
        
        root.calcExpectedReward()
        
        correctVals = []
        
        for child in root.children:
            if child != 0:
                correctVals.append(child.reward[root.state["currentPlayer"]])
            else:
                correctVals.append(0)
        
        return correctVals
        

class Trainer:
    
    PATH = "Agents/checkpoint"
    
    def __init__(self,logging_iteration, max_iteration=-1):
        
        self.logging_iteration = logging_iteration
        
        if max_iteration == -1:
            self.max_iteration = 9999999999999999
        else:
            self.max_iteration = max_iteration
            
    def trainTD(self,agentType):

        environment = Hearts()
        agent = agentType(environment.observation_space.n,environment.action_space.n)
        # if we have a saved model, continue training
        try:
            checkpoint = torch.load(self.PATH)
            iteration = checkpoint["iteration"] + 1
            agent.model.load_state_dict(checkpoint["model"])
            agent.optimizer.load_state_dict(checkpoint["optimizer"])

            losses = 0 
            averageReward = 0
            lossesList = checkpoint["losses"]
            averageRewardList = checkpoint["rewards"]

            print(f"Resuming training from iteration {iteration}")
        except:
            iteration = 1
            losses = 0 
            averageReward = 0
            lossesList = []
            averageRewardList = [] 
            
            
        memory = Memory(max_size=100000)
        loader = HeartsMemoryLoader()   
        
        for iteration in range(iteration, self.max_iteration + 1):
            done = False
            state = environment.reset()

            while not done:
                action, q = agent.act(state)
                isValid = environment.isActionValid(action)
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
                        loader.addReward(reward[i],i)
                        
                if not isValid:
                    loader.addReward(-1,currentPlayer)
                    
                state = next_state
                
            avgReward =  self.test(agent,RandomAgent(),RandomAgent(),RandomAgent(),1)
            averageReward += avgReward[0]
            

            for i in range(4):
                loader.loadNextState(state,i)
                memory.push(loader.ejectElement(i))
            
            memAmount = min(int(len(memory)**.5),64)
            
            for _ in range(memAmount):
                memory_batch = memory.get_batch(batch_size=memAmount)
                losses += agent.updateTD(memory_batch)/memAmount
            
            agent.update_randomness()

            if iteration % self.logging_iteration == 0:
                averageRewardList.append(averageReward/self.logging_iteration)
                lossesList.append(losses/self.logging_iteration)
                
                print(f"Iteration: {iteration}")
                print(f"  Loss: {(losses/self.logging_iteration):.5f}")
                print(f"  Average Reward: {(averageReward/self.logging_iteration):.5f}")
                print(f"  Average Reward of prev 10: {(sum(averageRewardList[-min(len(averageRewardList),10):])/min(len(averageRewardList),10)):.5f}")
                print()
                
                
                checkpoint = {
                    "iteration": iteration,
                    "model": agent.model.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "losses": lossesList,
                    "rewards": averageRewardList
                }
                
                torch.save(checkpoint, self.PATH)
                
                losses = 0
                averageReward = 0
                
        return lossesList, averageRewardList
    
    def trainMonte(self,agentType):
        environment = Hearts()
        agent = agentType(environment.observation_space.n,environment.action_space.n)
        # if we have a saved model, continue training
        try:
            checkpoint = torch.load(self.PATH)
            iteration = checkpoint["iteration"] + 1
            agent.model.load_state_dict(checkpoint["model"])
            agent.optimizer.load_state_dict(checkpoint["optimizer"])

            losses = 0 
            averageReward = 0
            lossesList = checkpoint["losses"]
            averageRewardList = checkpoint["rewards"]

            print(f"Resuming training from iteration {iteration}")
        except:
            iteration = 1
            losses = 0 
            averageReward = 0
            lossesList = []
            averageRewardList = [] 
            
        gameTree =  HeartGameTree(100)
        
        for iteration in range(iteration, self.max_iteration + 1):
            done = False
            state = environment.reset()

            while not done:
                
                saveState = environment.state
                
                correctVals = gameTree.search(state,environment,agent,agent.randomness)
                
                environment.state = saveState
                
                _, q = agent.act(state)
                
                losses += agent.updateMonte(q,correctVals)

                next_state, _, done = environment.step(np.argmax(correctVals))

                state = next_state
                
            avgReward =  self.test(agent,RandomAgent(),RandomAgent(),RandomAgent(),50)
            averageReward += avgReward[0]

            agent.update_randomness()

            if iteration % self.logging_iteration == 0:
                averageRewardList.append(averageReward/self.logging_iteration)
                lossesList.append(losses/self.logging_iteration)
                
                print(f"Iteration: {iteration}")
                print(f"  Loss: {(losses/self.logging_iteration):.5f}")
                print(f"  Average Reward: {(averageReward/self.logging_iteration):.5f}")
                print(f"  Average Reward of prev 10: {(sum(averageRewardList[-min(len(averageRewardList),10):])/min(len(averageRewardList),10)):.5f}")
                print()
                
                checkpoint = {
                    "iteration": iteration,
                    "model": agent.model.state_dict(),
                    "optimizer": agent.optimizer.state_dict(),
                    "losses": lossesList,
                    "rewards": averageRewardList
                }
                
                torch.save(checkpoint, self.PATH)
                
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
                isValid = environment.isActionValid(action)
                if not isValid:
                    totalRewards[currentPlayer] += -1
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
        
trainer = Trainer(1,100)
                
_, _ = trainer.trainMonte(Agent)           