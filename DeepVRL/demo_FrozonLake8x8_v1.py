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
parser.add_argument('--num_agents', default=100, type=int)  # 同时游走的agenet数量
parser.add_argument('--num_steps', default=100, type=int)  # 每个agent走的最大数量
parser.add_argument('--num_agents4eval', default=100, type=int)  # 同时游走的agenet数量
parser.add_argument('--num_steps4eval', default=50, type=int)  # 每个agent走的最大数量
parser.add_argument('--epoch', default=5000, type=int)  # 训练轮数
parser.add_argument('--lr', default=1e-3, type=float)  # 学习率
parser.add_argument('--k', default=10, type=int)  # 往前看的最大步长
args = parser.parse_args()

env_name = r"FrozenLake8x8-v1"
print("Env name: {}".format(env_name))
env = gym.make('FrozenLake8x8-v1', is_slippery=False)  # tinyGridWordld_root()
s_dim = env.observation_space.n  # 环境中的状态数目
a_dim = env.action_space.n  # 环境中的动作数目
# generate R and P tensor
# R = env.R
# P = env.P
num_of_agents = args.num_agents
num_of_steps = args.num_steps
num_of_agents4eval = args.num_agents4eval
num_of_steps4eval = args.num_steps4eval


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(nn.Embedding(s_dim, s_dim * 5),
                                   nn.Linear(s_dim * 5, 256),
                                   nn.Dropout(0.8),
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
net = Network()
net = net.to(device)
optimizer = optim.Adam(net.parameters(), lr=args.lr)
k = args.k
net.train()
gamma = 1.0  # 多步衰减因子
repaybuffer_s = []
repaybuffer_a = []
repaybuffer_r = []
repaybuffer_f = []
record = []
lam1 = 1.
lam2 = 1.
lam3 = 1.
thre = 2
for epoch in range(epochs):
    net.train()
    slist_, alist_, rlist_, flag_ = get_sample_traces_id(env, num_of_agents, num_of_steps)
    non_zero_index = torch.arange(len(rlist_.flatten()), dtype=torch.long)[rlist_.flatten() > 0]
    non_zero_index_trace = non_zero_index // num_of_steps
    repaybuffer_a.extend(alist_[non_zero_index_trace].flatten().tolist())
    repaybuffer_s.extend(slist_[non_zero_index_trace].flatten().tolist())
    repaybuffer_r.extend(rlist_[non_zero_index_trace].flatten().tolist())
    repaybuffer_f.extend(flag_[non_zero_index_trace].flatten().tolist())
    test = np.array(repaybuffer_a).reshape(-1, num_of_steps)
    if len(test) > thre:
        repaybuffer_f= np.array(repaybuffer_f)[- thre * num_of_steps:].tolist()
        repaybuffer_s= np.array(repaybuffer_s)[- thre * num_of_steps:].tolist()
        repaybuffer_r= np.array(repaybuffer_r)[- thre * num_of_steps:].tolist()
        repaybuffer_a = np.array(repaybuffer_a)[- thre * num_of_steps:].tolist()
    slist = torch.from_numpy(slist_).long().to(device).flatten()  # 状态列表
    alist = torch.from_numpy(alist_).long().to(device).flatten()  # 动作列表
    rlist = torch.from_numpy(rlist_).float().to(device).flatten()  # 奖励列表
    flag = torch.from_numpy(flag_).float().to(device).flatten()  # 每个agent的有效步数列表
    non_zero_index = torch.arange(len(rlist), device=device, dtype=torch.long)[rlist != 0]  # 获取有奖励的状态（步）
    # 考虑当前状态-奖励来计算目标函数H
    while not len(non_zero_index):  # 可能会出现所有agent都没有获得奖励
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
    non_zero_index = torch.arange(len(rlist), device=device, dtype=torch.long)[rlist > 0]
    non_zero_index_trace = non_zero_index // num_of_steps

    k_target = 0.
    # 从有奖励的轮次开始前推，计算前k步的H目标函数
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
    repay_target = 0.
    if (len(repaybuffer_a)):
        # print(repaybuffer_r)
        rlist = torch.from_numpy(np.array(repaybuffer_r, dtype=object).flatten().astype(np.float64)).to(device)
        slist = torch.from_numpy(np.array(repaybuffer_s, dtype=object).flatten().astype(np.int64)).to(device)
        alist = torch.from_numpy(np.array(repaybuffer_a, dtype=object).flatten().astype(np.int64)).to(device)
        flist = torch.from_numpy(np.array(repaybuffer_f, dtype=object).flatten().astype(np.float64)).to(device)
        non_zero_index = torch.arange(len(rlist), device=device, dtype=torch.long)[rlist > 0]
        non_zero_index_trace = non_zero_index // num_of_steps
        # 从有奖励的轮次开始前推，计算前k步的H目标函数
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
            repay_target -= (gamma ** ck) * torch.mean(out_pa * crlist)
    target = lam1 * first_target + lam2 * k_target + lam3 * repay_target
    # print(-target)
    optimizer.zero_grad()
    target.backward()
    optimizer.step()
    Pbar.update(epoch)
    print()
    net.eval()
    reward = evaluate_policy_by_policynet(env, net, num_of_agents4eval, num_of_steps4eval, device)
    record.append(reward)
    print('Total:{} TOP-1:{} TOP-2:{} TOP-3:{} Reward:{}'.format(target, first_target, k_target, repay_target,
                                                                 np.sum(reward)))

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
with open('ours_{}.pkl'.format(env_name), 'wb') as file:
    import pickle as pkl

    pkl.dump(record, file)
