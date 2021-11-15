import torch
import numpy as np
import gym

def evaluate_for_frozenlake(env, agent, num_agnet, num_steps, max_start_up_steps=10):
    scores = []
    for agent_i in range(num_agnet):
        flag = True
        done = False
        """
        while flag:
            start_up = np.random.choice(max_start_up_steps)
            state = env.reset()
            for start_up_i in range(start_up):
                action = np.random.choice(env.action_space.n)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                if done:
                    break
            if not done:
                flag = False"""
        sum_reward = 0.
        state = env.reset()
        for t in range(num_steps):
            action = agent.act(state, 0.01)
            next_state, reward, done, _ = env.step(action)
            sum_reward += reward
            if done:
                break
            state = next_state
        scores.append(sum_reward)
    return scores

def evaluate_for_gridworld(env, agent, num_agnet, num_steps, max_start_up_steps=10):
    scores = []
    for agent_i in range(num_agnet):
        flag = True
        done = False
        """
        while flag:
            start_up = np.random.choice(max_start_up_steps)
            state = env.reset()
            for start_up_i in range(start_up):
                action = np.random.choice(env.action_space.n)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                if done:
                    break
            if not done:
                flag = False"""
        sum_reward = 0.
        state = env.reset()
        for t in range(num_steps):
            action = agent.act(state, 0.)
            next_state, reward, done, _ = env.step(action)
            sum_reward += reward
            if done:
                break
            state = next_state
        scores.append(sum_reward)
    return scores

class littleWorld(gym.Env):
    def __init__(self):
        super().__init__()
        self.env_name = "littleWorld"
        self.observation_space = gym.spaces.discrete.Discrete(100)
        self.action_space = gym.spaces.discrete.Discrete(4)
        self.R, _ = self.initialize_R(100, 4)
        self.P, _ = self.initialize_P(100, 4)
        self.state = 0
        self.reward_range = (-10, 20)
        self.metadata = {}
        self.state_dim = 100
        self.action_dim = 4
        self.trigger = False

    def step(self, action):
        reward = self.R[self.state, action].item()
        self.state = torch.argmax(self.P[self.state, action]).item()
        done = False
        # if self.state in [9, 19, 29, 39]:
            # done = True
        return self.state, reward, done, ""

    def reset(self):
        self.state = np.random.randint(self.observation_space.n)
        # self.state = 0
        return self.state

    """
    0 0 0 0 0 0 0 0 0 0 0 9
    1 0 0 0 1 0 0 0 0 0 0 19
    0 0 0 0 0 0 0 0 0 0 0 29
    0 0 0 0 1 0 0 0 0 0 0 39
    0 0 0 0 0 0 0 0 0 0 0 49
    0 0 0 0 0 0 0 0 0 0 0 59
    0 0 0 0 0 0 0 0 0 0 0 69
    0 0 0 0 0 0 0 0 0 0 0 79
    0 0 0 0 0 0 0 0 0 0 0 89
    0 0 0 0 0 0 0 0 0 0 1 99
    """

    def initialize_R(self, s, a):
        R = torch.zeros((s, a))
        R[s // 3, :] = 10
        R[s // 10, :] = 20
        R[s // 7, :] = 30
        R[s - 1, :] = 40
        R[s // 2, :] = -100
        for i in range(s):
            if i % 10 == 0:
                R[i, 0] = -10
            if (i - 9) % 10 == 0:
                R[i, 1] = -10
            if i in range(10):
                R[i, 2] = -10
            if i in range(s - 10, s):
                R[i, 3] = -10
        R_vec = R.reshape(s * a)
        return R, R_vec

    def initialize_P(self, s, a):
        print('get P')
        P = torch.zeros((s, a, s))  # s_t, a, s_(t+1)
        for i in range(s):
            if i % 10 == 0:
                P[i, 0, i] = 1
            if (i - 9) % 10 == 0:
                P[i, 1, i] = 1
            if i in range(10):
                P[i, 2, i] = 1
            if i in range(s - 10, s):
                P[i, 3, i] = 1

            for j in range(s):
                if j == i - 1:
                    P[i, 0, j] = 1
                if j == i + 1:
                    P[i, 1, j] = 1
                if j == i - 10:
                    P[i, 2, j] = 1
                if j == i + 10:
                    P[i, 3, j] = 1
        P[s // 3, :, s // 3 + 8] = 1
        P[s // 10, :, s // 10 - 4] = 1
        P[s // 7, :, s // 7 + 3] = 1
        P[s - 1, :, s - 7] = 1
        P_mat = torch.empty((s, a, s, a))
        for ss in range(s):
            for i in range(a):
                P_mat[:, :, ss, i] = P[:, :, ss]
        P_mat = P_mat.reshape(s * a, s * a)
        return P, P_mat
        