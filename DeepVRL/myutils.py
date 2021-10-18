import torch
import torch.optim as optim

import numpy as np
import pkbar
import time

import tensorly as tl
import pycuda.autoinit
import pycuda
from pycuda import compiler
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
from tensorly.decomposition import tensor_train, parafac
from tensorly.tt_tensor import tt_to_tensor
tl.set_backend('pytorch')
import tensornetwork as tn
from tensornetwork import contractors
from numba import jit

def softmax_by_state(data, s_size, a_size): # 将Q表概率化
    states = []
    softmax = torch.nn.Softmax(dim=0)
    for s in range(s_size):
        state = data[s * a_size: (s + 1) * a_size, :]
        states.append(softmax(state))
    cat = torch.cat(states, dim=0)
    return cat


def get_sample_traces_id(env, num_agent, num_steps): # 探索环境
    slist = []
    alist = []
    rlist = []
    flag = []
    for a in range(num_agent):
        state = env.reset()
        if type(state) == int:
            state = [state]
        # print(state)
        # exit()
        agent_slist = [[-1 for _ in range(len(state))] for _ in range(num_steps)]
        agent_rlist = [0. for _ in range(num_steps)]
        agent_alist = [0 for _ in range(num_steps)]
        agent_flag = [0 for _ in range(num_steps)]
        for k in range(num_steps):
            action = np.random.choice(env.action_space.n)
            next_state, reward, done, info = env.step(action)
            agent_rlist[k] = reward
            agent_slist[k][:] = state[:]
            agent_alist[k] = action
            agent_flag[k] = 1
            if done:
                if agent_rlist[k] == 0 and k != num_steps-1:  # 提前因陷阱而终止
                    agent_rlist[k] = -10
                break
            else:
                if next_state == state:  # 非法行动 状态不变
                    if not agent_rlist[k]:
                        agent_rlist[k] = -10
                state = next_state
                if type(state) == int:
                    state = [state]
        rlist.append(agent_rlist)
        slist.append(agent_slist)
        alist.append(agent_alist)
        flag.append(agent_flag)
    return np.array(slist), np.array(alist), np.array(rlist), np.array(flag)
