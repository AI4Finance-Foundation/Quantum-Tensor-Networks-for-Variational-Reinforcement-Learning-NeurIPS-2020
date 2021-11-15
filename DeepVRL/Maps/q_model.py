import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import gym
import matplotlib.pyplot as plt

class DiscreteQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DiscreteQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.fc = nn.Sequential(
            nn.Embedding(state_size,state_size*3),
            nn.Linear(state_size*3, 128),
            nn.ReLU(),
            nn.Linear(128, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.fc(state)
class ContinueQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(DiscreteQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        "*** YOUR CODE HERE ***"
        self.fc = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, action_size),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        return self.fc(state)


def pre_process(observation):
    """Process (210, 160, 3) picture into (1, 84, 84)"""
    x_t = cv2.cvtColor(cv2.resize(observation, (84, 84)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
    return np.reshape(x_t, (1, 84, 84)), x_t


def stack_state(processed_obs):
    """Four frames as a state"""
    return np.stack((processed_obs, processed_obs, processed_obs, processed_obs), axis=0)


if __name__ == '__main__':
    env = gym.make('Breakout-v0')
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', env.action_space.n)

    obs = env.reset()
    x_t, img = pre_process(obs)
    state = stack_state(img)
    print(np.shape(state[0]))

    state = torch.randn(32, 4, 84, 84)  # (batch_size, color_channel, img_height,img_width)
    state_size = state.size()

    cnn_model = ContinueQNetwork(state_size, action_size=4, seed=1)
    outputs = cnn_model(state)
    print(outputs)