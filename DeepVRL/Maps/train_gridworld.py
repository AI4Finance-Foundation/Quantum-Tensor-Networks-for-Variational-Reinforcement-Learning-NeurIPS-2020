import gym
import random
import torch
import numpy as np
from collections import deque
from dqn_agent import Agent
import matplotlib.pyplot as plt
import cv2
import time
import MyEnv
import time
import argparse
import pkbar

parser = argparse.ArgumentParser()
parser.add_argument('--num_agents', default=10000, type=int)  # 同时游走的agenet数量
parser.add_argument('--num_steps', default=1000, type=int)  # 每个agent走的最大数量
parser.add_argument('--num_agents4eval', default=100, type=int)  # 同时游走的agenet数量
parser.add_argument('--num_steps4eval', default=100, type=int)  # 每个agent走的最大数量
parser.add_argument('--epoch', default=1000, type=int)  # 训练轮数
parser.add_argument('--lr', default=1e-3, type=float)  # 学习率
parser.add_argument('--opt',default='t',type=str) # 用哪种优化方式 q/h/a/t q:Q-Learning h:Hamiltonian a:Q+H t:Tree
args = parser.parse_args()




env = gym.make('Taxi-v3') # gym.make('Taxi-v3')# MyEnv.littleWorld()
state_size = env.observation_space.n
action_size = env.action_space.n
print('Original state shape: ', state_size)
print('Number of actions: ', env.action_space.n)

agent = Agent(state_size, action_size, seed=1)  # state size (batch_size, 4 frames, img_height, img_width)
TRAIN = True  # train or test flag 
if args.opt == 'q':
    print("only use q learning")
elif args.opt == 'h':
    print("only use Hamiltonian learning")
    agent.heat.RandomWalk(env, args.num_agents, args.num_steps)
elif  args.opt == 'a':
    print("use Q and Hamiltonian learning")
    agent.heat.RandomWalk(env, args.num_agents, args.num_steps)
elif  args.opt == 't':
    print("use Hamiltonina Tree")
    agent.tree.RandomWalk(env, args.num_agents, args.num_steps)




def dqn(n_episodes=60, max_t=400, eps_start=0.1, eps_end=0.1, eps_decay=0.9995):
    """Deep Q-Learning.
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode, maximum frames
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=10)  # last 100 scores
    eps = eps_start  # initialize epsilon
    merge_results = []
    used_time = 0.
    for i_episode in range(1, n_episodes + 1):
        episode_start = time.time()
        state = env.reset()
        state_traj = []
        action_traj = []
        reward_traj = []
        single_reward_traj = []
        score = 0
        for t in range(max_t):
            if args.opt == 't':
                break 
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            if args.opt == 'a' or args.opt == 'q':
                agent.step(state, action, reward, next_state, done)
            score += reward
            state_traj.append(state)
            action_traj.append(action)
            reward_traj.append(score)
            single_reward_traj.append(reward)
            state = next_state
            if done:
                break
        if args.opt == 'a' or args.opt == 'h':
            agent.step_traj(state_traj, action_traj, reward_traj)
        elif args.opt == 't':
            agent.step_traj_in_Tree(state_traj, action_traj, single_reward_traj)
        episode_end = time.time()
        used_time = episode_end - episode_start
        result = MyEnv.evaluate_for_gridworld(env, agent, args.num_agents4eval, args.num_steps4eval, max_start_up_steps=5)
        merge_results.append([i_episode, used_time, score])
        score = np.mean(result)
        scores_window.append(score)  # save most recent score
        scores.append(score)  # save most recent score
        eps = max(eps_end, eps_decay * eps)  # decrease epsilon
        # print('\tEpsilon now : {:.2f}'.format(eps))
        print('Episode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
        if i_episode % 1000 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            print('\rEpisode {}\tThe length of replay buffer now: {}'.format(i_episode, len(agent.memory)))
    import os
    if not os.path.exists('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(agent.qnetwork_local.state_dict(), 'dqn_checkpoint_8.pth')
    import pickle as pkl
    with open('result/method-{}-env-{}.pkl'.format(args.opt,'gw'),'wb') as file:
        pkl.dump(merge_results, file)
    return scores


if __name__ == '__main__':
    if TRAIN:
        start_time = time.time()
        scores = dqn()
        print('COST: {} min'.format((time.time() - start_time)/60))
        print("Max score:", np.max(scores))

        # plot the scores
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.plot(np.arange(len(scores)), scores)
        # plt.ylabel('Score')
        # plt.xlabel('Episode #')
        # plt.show()

    else:
        # load the weights from file
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint/dqn_checkpoint_8.pth'))
        rewards = []
        for i in range(10):  # episodes, play ten times
            total_reward = 0
            state = env.reset()
            for j in range(10000):  # frames, in case stuck in one frame
                action = agent.act(state)
                env.render()
                next_state, reward, done, _ = env.step(action)
                total_reward += reward

                # time.sleep(0.01)
                if done:
                    rewards.append(total_reward)
                    break

        print("Test rewards are:", *rewards)
        print("Average reward:", np.mean(rewards))
        env.close()