import torch
import numpy as np

class HeartsNN(torch.nn.Module):
    def __init__(self, observation_space, action_space, hidden_shape):
        super(HeartsNN, self).__init__()

        self.inputLayer = torch.nn.Linear(observation_space, hidden_shape[0])

        self.hiddenLayers = HiddedLayers(hidden_shape)

        self.outputLayer = torch.nn.Linear(hidden_shape[-1], action_space)

    def forward(self, x):
        x = self.inputLayer(x)
        x = torch.nn.functional.leaky_relu(x)

        x = self.hiddenLayers.forward(x)
        x = self.outputLayer(x)

        return x

class HiddedLayers(torch.nn.Module):
    def __init__(self, shape):
        super(HiddedLayers, self).__init__()

        self.shape = shape

        for i in range(len(shape)-1):
            setattr(self, f"hidden_{i}", torch.nn.Linear(shape[i], shape[i+1]))
    
    def forward(self, x):
        
        for i in range(len(self.shape)-1):
            x = getattr(self, f"hidden_{i}")(x)
            x = torch.nn.functional.leaky_relu(x)

        return x


class Agent(object):
    def __init__(self, inputs, outputs):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = HeartsNN(inputs, outputs, [512,256,126,64,32,16]).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=.0005)

        self.decay = 0.995
        self.randomness = 1.00
        self.min_randomness = 0.0001

    def act(self, state):
        # move the state to a Torch Tensor
        state = torch.tensor(state).float().to(self.device)

        # find the quality of both actions
        qualities = self.model(state).cpu()

        # sometimes take a random action
        if np.random.rand() <= self.randomness:
            action = np.random.randint(low=0, high=qualities.size(dim=0))
        else:
            action = torch.argmax(qualities).item()

        # return that action
        return action, qualities

    def update(self, memory_batch):
        # unpack our batch and convert to tensors
        states, next_states, actions, rewards = self.unpack_batch(memory_batch)

        # compute what the output is (old expected qualities)
        # Q(S, A)
        old_targets = self.old_targets(states, actions)

        # compute what the output should be (new expected qualities)
        # reward + max_a Q(S', a)
        new_targets = self.new_targets(next_states, rewards)

        # compute the difference between old and new estimates
        loss = torch.nn.functional.smooth_l1_loss(old_targets, new_targets)

        # apply difference to the neural network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # for logging
        return loss.item()

    def old_targets(self, states, actions):
        # model[states][action]
        actions = actions.type(torch.int64)
        return self.model(states).gather(1, actions)

    def new_targets(self, next_states, rewards):
        # reward + max(model[next_state])
        return rewards + torch.amax(self.model(next_states), dim=1, keepdim=True)

    def unpack_batch(self, batch):
        states, actions, next_states, rewards = zip(*batch)

        states = torch.tensor(np.array(states)).float().to(self.device)
        next_states = torch.tensor(np.array(next_states)).float().to(self.device)

        # unsqueeze(1) makes 2d array. [1, 0, 1, ...] -> [[1], [0], [1], ...]
        # this is required because the first array is for the batch, and
        #   the inner arrays are for the elements
        # the states and next_states are already in this format so we don't
        #   need to do anything to them
        # .long() for the actions because we are using them as array indices
        actions = torch.tensor(np.array(actions)).long().unsqueeze(1).to(self.device)
        rewards = torch.tensor(np.array(rewards)).float().unsqueeze(1).to(self.device)

        return states, next_states, actions, rewards

    def update_randomness(self):
        self.randomness *= self.decay
        self.randomness = max(self.randomness, self.min_randomness)
        
        
        
        
class RandomAgent(object):

    def act(self, state):
        return 0, []