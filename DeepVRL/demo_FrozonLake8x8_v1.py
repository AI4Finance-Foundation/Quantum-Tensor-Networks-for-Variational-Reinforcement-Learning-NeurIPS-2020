import gym
import os
import numpy as np
import torch
from VRL import *
import seaborn as sns
import matplotlib.pyplot as plt
from tensorly.tenalg import inner, outer
from gym_toytext import *
from Envs import tinyGridWord, tinyGridWordld_root
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import argparse
import pkbar

parser = argparse.ArgumentParser()
parser.add_argument('--num_agents', default=1000, type=int) # 同时游走的agenet数量
parser.add_argument('--num_steps', default=100, type=int) # 每个agent走的最大数量
parser.add_argument('--epoch', default=1000, type=int) # 训练轮数
parser.add_argument('--lr', default=1e-3, type=int) # 学习率
parser.add_argument('--k', default=20, type=int) # 往前看的最大步长
args = parser.parse_args()

env_name = r"FrozenLake8x8-v1"
print("Env name: {}".format(env_name))
env = gym.make('FrozenLake8x8-v1', is_slippery=False)  # tinyGridWordld_root()
s_dim = env.observation_space.n
a_dim = env.action_space.n
# generate R and P tensor
# R = env.R
# P = env.P
num_of_agents = args.num_agents
num_of_steps = args.num_steps


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Embedding(s_dim, s_dim * 5),
                                   nn.Linear(s_dim * 5, 256),
                                   nn.Dropout(0.6),
                                   nn.ReLU(),
                                   nn.Linear(256, 4),
                                   nn.Softmax(dim=1))

    def forward(self, x):
        y = self.model(x)
        return y

epochs = args.epoch
device = "cuda:0" if torch.cuda.is_available() else "cpu"



print("")
Pbar = pkbar.Pbar(name='Training progress of our policy', target=epochs)
energy_history = []
net = Network(input_size=2, max_steps=20, hidden_size=64, num_layer=2)
net = net.to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
k = args.k
net.train()
gamma = 1.0  # 多步衰减因子
for epoch in range(epochs):
    slist, alist, rlist, flag = get_sample_traces_id(env, num_of_agents, num_of_steps)
    slist = torch.from_numpy(slist).long().to(device).flatten()
    alist = torch.from_numpy(alist).long().to(device).flatten()
    rlist = torch.from_numpy(rlist).float().to(device).flatten()
    flag = torch.from_numpy(flag).float().to(device).flatten()
    non_zero_index = torch.arange(len(rlist), device=device, dtype=torch.long)[rlist != 0]
    # 考虑当前状态-动作损失
    while not len(non_zero_index):
        slist, alist, rlist, flag = get_sample_traces_id(env, num_of_agents, num_of_steps)
        slist = torch.from_numpy(slist).long().to(device).flatten()
        alist = torch.from_numpy(alist).long().to(device).flatten()
        rlist = torch.from_numpy(rlist).float().to(device).flatten()
        flag = torch.from_numpy(flag).float().to(device).flatten()
        non_zero_index = torch.arange(len(rlist), device=device, dtype=torch.long)[rlist != 0]
    cslist = slist[non_zero_index]
    calist = alist[non_zero_index]
    crlist = rlist[non_zero_index]
    onehot = cslist  # torch.zeros(size=[len(non_zero_index), s_dim], device=device).scatter_(1, cslist.reshape(-1, 1), 1)
    out = net(onehot)
    out_pa = out[np.arange(len(out)), calist.flatten()]
    first_target = - torch.mean(out_pa * crlist)
    non_zero_index = torch.arange(len(rlist), device=device, dtype=torch.long)[rlist> 0]
    k_target = 0.
    #从有奖励的轮次开始前推，计算k步的H目标函数
    for ck in range(1, k + 1):
        pre_k_step = non_zero_index - ck
        pre_k_step = pre_k_step[pre_k_step >= 0]
        valid_pre_k_step = pre_k_step[flag[pre_k_step] > 0]
        if not len(valid_pre_k_step):
            break
        out_pa = 1.
        for ck_i in range(ck + 1):
            cslist = slist[valid_pre_k_step + ck_i]
            calist = alist[valid_pre_k_step + ck_i]
            onehot = cslist  # torch.zeros(size=[len(cslist), s_dim], device=device).scatter_(1, cslist.reshape(-1, 1), 1)
            out = net(onehot)
            out_pa *= out[np.arange(len(out)), calist.flatten()]
        crlist = rlist[valid_pre_k_step + ck]
        k_target -= (gamma ** ck) * torch.mean(out_pa * crlist)
    target = first_target + k_target
    # print(-target)
    optimizer.zero_grad()
    target.backward()
    optimizer.step()
    Pbar.update(epoch)
    print(-target.detach().cpu().numpy())

# retrieve the resulting policy
net.eval()
net = net.to('cpu')
u = []
for s in range(env.observation_space.n):
    hot = s  # np.zeros(shape=s_dim)
    # hot[s] = 1
    u.append(hot)
u = np.array(u)
u = torch.from_numpy(u).long()
p = net(u)
p = p.reshape((env.observation_space.n, env.action_space.n))
our_pi = p
our_result = our_pi.detach().cpu().numpy().reshape(s_dim, a_dim)
our_policy = torch.empty(s_dim)
p = torch.tensor(our_result)
for i in range(s_dim):
    our_policy[i] = torch.argmax(p[i, :])
# our_policy = our_policy.reshape(s_dim, s_dim)
# print(our_policy)

print("\n-------------------Result-------------------")

print("\n- Our Policy:")
print(our_policy.reshape((8, 8)))
