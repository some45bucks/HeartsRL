import numpy as np
from HeartsGym import Hearts
from Agent import Agent
from Memory import Memory, HeartsMemoryLoader


max_iteration = 1
logging_iteration = 2

environment = Hearts()
agent = Agent(environment.observation_space.n,environment.action_space.n)
memory = Memory(max_size=10000)
loader = HeartsMemoryLoader()

for iteration in range(1, max_iteration + 1):
    done = False
    state = environment.reset()

    roundReward = []

    while not done:
        action, q = agent.act(state)
        currentPlayer = environment.getCurrentPlayer()

        environment.render()

        cv = "Action: {:5s}  ".format(str(action))

        for x in q:
            cv += "{:7s}".format("{:.2f}".format(x.item()))

        print(cv)

        exit = input("Continue ")

        if exit:
            break

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

    environment.render()

    for i in range(4):
        memory.push(loader.ejectElement(i))

    # for _ in range(64):
    #     memory_batch = memory.get_batch(batch_size=64)
    #     loss = agent.update(memory_batch)
    
    # losses.append(loss)
    agent.update_randomness()

    if iteration % logging_iteration == 0:
        print(f"Iteration: {iteration}")
        print(f"  Memory-Buffer Size: {len(memory.memory)}")
        print(f"  Agent Randomness: {agent.randomness:.3f}")
        print()