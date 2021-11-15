from os import EX_OK
from typing import NoReturn
import numpy as np
import random
from collections import namedtuple, deque
from cv2 import mulSpectrums

from q_model import DiscreteQNetwork as QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

# from torch._C import preserve_format
torch.autograd.set_detect_anomaly(True)
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 100         # minibatch size
GAMMA = 0.99           
TAU = 1e-3              # for soft update of target parameters
LR = 1e-3             # learning rate
UPDATE_EVERY = 1        # how often to update the network
K = 15                    # The maximize step backward
print(K)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cpu'
print(device)

class DiscreteMap:
    def __init__(self, num_root, num_branch):
        self.container = {}
        self.num_root = {}
        self.max_num_root = num_root
        self.max_num_branch = num_branch
    def append(self, state_list, action_list, reward_list, k):
        reward_list = np.array(reward_list)
        index_step = np.where(reward_list>0)[0]
        for step in index_step:
            selected_state = state_list[step]
            if selected_state not in list(self.container.keys()):
                self.container[selected_state] = {'reward':reward_list[step],'braches_state':[],'braches_action':[]}
                if len(self.container.keys())>self.max_num_root:
                    min_reward_root = None
                    min_reward = np.inf
                    for key in self.container.keys():
                        if self.container[key]['reward'] < min_reward:
                            min_reward = self.container[key]['reward']
                            min_reward_root = key
                    self.container.pop(min_reward_root)
            branch_state = []
            branch_action = []
            for pre_step in range(step-1, step-1-k,-1):
                if pre_step == -1:
                    break
                branch_state.append(state_list[pre_step])
                branch_action.append(action_list[pre_step])
            self.container[selected_state]['braches_state'].append(branch_state)
            self.container[selected_state]['braches_action'].append(branch_action)
            if len(self.container[selected_state]['braches_state'])>self.max_num_branch:
                self.container[selected_state]['braches_state'].pop(0)
                self.container[selected_state]['braches_action'].pop(0)
    def __len__(self):
        return len(self.container.keys())
    
    def sample(self, batch_size):
        batch_state = []
        batch_action = []
        batch_reward = []
        num = 0.
        d = list(self.container.keys())
        if len(d)==0:
            return [], [], []
        while num < batch_size:
            select_d = np.random.choice(d, size=1)[0]
            to_select_state  = self.container[select_d]['braches_state']
            to_select_action = self.container[select_d]['braches_action']
            to_select_reward = self.container[select_d]['reward']
            len_to_select = len(to_select_action)
            select_idx = np.random.choice(len_to_select, size=1)[0]
            batch_state.append(to_select_state[select_idx])
            batch_action.append(to_select_action[select_idx])
            batch_reward.append(to_select_reward)
            num += len(to_select_state[select_idx])
        return batch_state, batch_action, batch_reward


        


class HeatMap:
    def __init__(self, action_size, num_root, num_branch, batch_size, seed):
        self.action_size = action_size
        self.batch_size = batch_size
        self.memory = DiscreteMap(num_root, num_branch)
        self.seed = random.seed(seed)
    def add(self,  state_list, action_list, reward_list, max_k):
        """Add a new experience to memory."""
        self.memory.append(state_list, action_list, reward_list, max_k)
    
    def sample(self, batch_size=BATCH_SIZE):
        """Randomly sample a batch of experiences from memory."""
        batch_state, batch_action, batch_reward = self.memory.sample(batch_size)
        return batch_state, batch_action, batch_reward

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
    def RandomWalk(self, env, num_agent, num_step, max_K = K):
        for agent in range(num_agent):
            have_steps = 0
            sum_reward = 0.
            state_agent = []
            action_agent = []
            reward_agent = []
            state = env.reset()
            for t in range(num_step):
                action = np.random.choice(self.action_size)
                next_state, reward, done, _ = env.step(action)
                state_agent.append(state)
                action_agent.append(action)
                sum_reward += reward
                reward_agent.append(sum_reward)
                have_steps += 1
                if have_steps >= num_step:
                    break
                if done:
                    break
                state = next_state
            self.add(state_agent,action_agent,reward_agent,max_K)
    



class DiscreteGenerateForest:
    def __init__(self, state_size, action_size, batch_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.u_size = self.state_size * self.action_size
        self.batch_size = batch_size
        self.u_reward = torch.from_numpy(np.array([0 for _ in range(self.u_size)])) #每个u获得的奖励
        self.maps = [ None for _ in range(self.u_size)]
        self.pre_maps = [None for _ in range(self.u_size)]
        self.maps_indicate = []
        self.seed = seed

    def dijkstra(self, graph, s, dis_threshold=np.inf):
        # print('in dij')
        # print(s)
        # print('-----')
        n = len(graph)
        # 创建一个flag数组，用于保存遍历情况
        flag = np.zeros(shape=(n,), dtype=bool)
        # 创建一个dist数组，用于保存最短路径
        dis = graph[s]
        # print("distance")
        # print(dis)
        # print(dis[s])
        # print("-------")
        # 创建一个prev数组，用于保存对应的最短节点
        prev = np.zeros(shape=(n,), dtype=int) - 1
        prev[np.where(dis<np.inf)[0]] = s
        # 将源节点放入集合S中
        flag[s] = True
        # print(graph)
        # Dijkstra算法中重点：
        # 迭代次数为n-1次，因为如果确定某一节点，但其最小值不会影响其他节点，每次迭代只能确定一个节点；
        # 依次将节点放入集合S中（即已访问过的节点）；
        for i in range(n-1):
            # 找到当前dist中还未被遍历的节点中权值最小的节点；
            # 并将其放入集合S中；
            temp = float('inf')
            u = 0
            for j in range(n):
                if not flag[j] and dis[j] != float('inf') and dis[j] < temp:
                    u = j
                    temp = dis[j]
            if dis[u] > dis_threshold:
                break
            flag[u] = True
            # 确定当前节点最短距离后，其他节点最短距离是否随之改变，若改变，即可确定其最短路径；
            for j in range(n):
                if not flag[j] and graph[u][j] != float('inf'):
                    if dis[u] + graph[u][j] < dis[j] or dis[j] == float('inf'):
                        dis[j] = dis[u] + graph[u][j]
                        prev[j] = u
        # 输出结果
        return prev
    
    def prev_to_map(self, prev_list):
        map = np.ones(shape=[len(prev_list), len(prev_list)])*np.inf
        for u in range(len(prev_list)):
            if prev_list[u] == -1:
                continue
            else:
                map[prev_list[u],u]=1
        return map

    def find_dominator(self, graph: np.ndarray, parks: np.ndarray, K: int):
        num_nodes = len(graph)
        dominators = np.zeros(num_nodes) - 1
        dominators[parks] = parks
        last_update_list = list(parks)
        for dis in range(1, K+1):
            new_update_list = []
            for node in last_update_list:
                adj_nodes = np.where(graph[node] > 0)[0]
                for adj_node in adj_nodes:
                    if dominators[adj_node] == -1:
                        dominators[adj_node] = dominators[node]
                        new_update_list.append(adj_node)
            last_update_list[:] = new_update_list[:]
        return dominators

    def build(self, state_list, action_list, reward_list, max_k=K):
        # 给定多条轨迹，其中reward是单步u的奖励，不是累计奖励，构造最小生成树对应的邻接矩阵和邻接表
        # self.maps[plus_u]： 以plus_u为热源的生成树
        # self.pre_maps[plus_u]： 以plus_u为热源的生成树的前驱列表 一维度
        # self.maps_indicate： 存储了有哪些根
        # print(len(state_list))
        for traj_index in range(len(state_list)):
            state = np.array(state_list[traj_index])
            action = np.array(action_list[traj_index])
            reward = np.array(reward_list[traj_index])
            u = state * self.action_size + action
            for j in range(len(u)):
                self.u_reward[u[j]] = reward[j]
            reward_index = np.where(reward>0)[0]
            # print(reward_index)
            for index in reward_index:
                pre_K = max(0, index-K)
                # print(pre_K)
                if self.maps[u[index]] is None:
                    self.maps[u[index]] = np.ones(shape=[self.u_size, self.u_size]) * np.inf
                    self.maps_indicate.append(u[index])
                for inner in range(index, pre_K, -1):
                    if u[inner] == u[inner-1]:
                        continue
                    if self.u_reward[u[inner-1]]>0:
                        break
                    # assert not ((u[inner] == 0) and (u[inner-1] == 3)), "old method there should be path from 0 to 3 "
                    # if (u[inner] == 0) and (u[inner-1] == 3):
                    #     print(u[index])
                    #     exit(0)
                    self.maps[u[index]][u[inner],u[inner-1]] = 1
        print(self.maps[58][0, 3])
        for reward_u in self.maps_indicate:
            # print(reward_u)
            self.pre_maps[reward_u] = self.dijkstra(self.maps[reward_u], reward_u)
            self.maps[reward_u] = self.prev_to_map(self.pre_maps[reward_u])
            # print('------')
            # print(reward_u)
            # print(self.pre_maps[reward_u])
            # print(self.maps[reward_u][reward_u])
        # exit()
        print(self.maps[58][0, 3])
        # exit(0)
        return 
        
    def RandomWalk(self, env, num_agent, num_step, max_K = K):
        state_list = []
        action_list = []
        reward_list  =[]
        # print(num_agent)
        for agent in range(num_agent):
            state_agent = []
            action_agent = []
            reward_agent = []
            state = env.reset()
            for t in range(num_step):
                action = np.random.choice(self.action_size)
                next_state, reward, done, _ = env.step(action)
                # if reward:
                    # print('+++++')
                state_agent.append(state)
                action_agent.append(action)
                reward_agent.append(reward)
                if done:
                    break
                state = next_state
            state_list.append(state_agent)
            action_list.append(action_agent)
            reward_list.append(reward_agent)
        # print('--------')
        # print(len(state_list))
        # print("Successful random walk")
        self.build(state_list, action_list, reward_list, max_K)
        return state_list, action_list, reward_list, max_K
                

class newDiscreteGenerateForest:
    def __init__(self, state_size, action_size, batch_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.u_size = self.state_size * self.action_size
        self.batch_size = batch_size
        self.u_reward = torch.from_numpy(np.array([0 for _ in range(self.u_size)])) #每个u获得的奖励
        self.maps = [ None for _ in range(self.u_size)]
        self.pre_maps = [None for _ in range(self.u_size)]
        self.maps_indicate = []
        self.seed = seed
        self.graph = np.zeros((self.u_size, self.u_size)) + np.inf  # the global ajdacent matrix

    def prev_to_map(self, prev_list):
        map = np.ones(shape=[len(prev_list), len(prev_list)])*np.inf
        for u in range(len(prev_list)):
            if prev_list[u] == -1:
                continue
            else:
                map[prev_list[u],u]=1
        return map

    def dijkstra(self, graph, s, dis_threshold=np.inf):
        # print('in dij')
        # print(s)
        # print('-----')
        n = len(graph)
        # 创建一个flag数组，用于保存遍历情况
        flag = np.zeros(shape=(n,), dtype=bool)
        # 创建一个dist数组，用于保存最短路径
        dis = graph[s].copy()
        # print("distance")
        # print(dis)
        # print(dis[s])
        # print("-------")
        # 创建一个prev数组，用于保存对应的最短节点
        prev = np.zeros(shape=(n,), dtype=int) - 1
        prev[np.where(dis<np.inf)[0]] = s
        # 将源节点放入集合S中
        flag[s] = True
        # print(graph)
        # Dijkstra算法中重点：
        # 迭代次数为n-1次，因为如果确定某一节点，但其最小值不会影响其他节点，每次迭代只能确定一个节点；
        # 依次将节点放入集合S中（即已访问过的节点）；
        for i in range(n-1):
            # 找到当前dist中还未被遍历的节点中权值最小的节点；
            # 并将其放入集合S中；
            temp = float('inf')
            u = 0
            for j in range(n):
                if not flag[j] and dis[j] != float('inf') and dis[j] < temp:
                    u = j
                    temp = dis[j]
            if dis[u] > dis_threshold:
                break
            flag[u] = True
            # 确定当前节点最短距离后，其他节点最短距离是否随之改变，若改变，即可确定其最短路径；
            for j in range(n):
                if not flag[j] and graph[u][j] != float('inf'):
                    if dis[u] + graph[u][j] < dis[j] or dis[j] == float('inf'):
                        dis[j] = dis[u] + graph[u][j]
                        prev[j] = u
        # 输出结果
        return prev
    
    def prev_to_map(self, prev_list):
        map = np.ones(shape=[len(prev_list), len(prev_list)])*np.inf
        for u in range(len(prev_list)):
            if prev_list[u] == -1:
                continue
            else:
                map[prev_list[u],u]=1
        return map

    def build_trees(self, state_list, action_list, reward_list, max_k=K):
        # # 给定多条轨迹，其中reward是单步u的奖励，不是累计奖励，构造最小生成树对应的邻接矩阵和邻接表
        # # self.maps[plus_u]： 以plus_u为热源的生成树
        # # self.pre_maps[plus_u]： 以plus_u为热源的生成树的前驱列表 一维度
        # # self.maps_indicate： 存储了有哪些根
        # # print(len(state_list))
        # for traj_index in range(len(state_list)):
        #     state = np.array(state_list[traj_index])
        #     action = np.array(action_list[traj_index])
        #     reward = np.array(reward_list[traj_index])
        #     u = state * self.action_size + action
        #     for j in range(len(u)):
        #         self.u_reward[u[j]] = reward[j]
        #     reward_index = np.where(reward>0)[0]
        #     # print(reward_index)
        #     for index in reward_index:
        #         pre_K = max(0, index-K)
        #         # print(pre_K)
        #         if self.maps[u[index]] is None:
        #             self.maps[u[index]] = np.ones(shape=[self.u_size, self.u_size]) * np.inf
        #             self.maps_indicate.append(u[index])
        #         for inner in range(index, pre_K, -1):
        #             if u[inner] == u[inner-1]:
        #                 continue
        #             if self.u_reward[u[inner-1]]>0:
        #                 break
        #             self.maps[u[index]][u[inner],u[inner-1]] = 1
        # for reward_u in self.maps_indicate:
        #     # print(reward_u)
        #     self.pre_maps[reward_u] = self.dijkstra(self.maps[reward_u], reward_u)
        #     self.maps[reward_u] = self.prev_to_map(self.pre_maps[reward_u])
        #     # print('------')
        #     # print(reward_u)
        #     # print(self.pre_maps[reward_u])
        #     # print(self.maps[reward_u][reward_u])
        # # exit()
        # return 
        pass

    def RandomWalk(self, env, num_agent, num_step, max_K = K):
        state_list = []
        action_list = []
        reward_list  =[]
        # print(num_agent)
        for agent in range(num_agent):
            state_agent = []
            action_agent = []
            reward_agent = []
            state = env.reset()
            for t in range(num_step):
                action = np.random.choice(self.action_size)
                next_state, reward, done, _ = env.step(action)
                # if reward:
                    # print('+++++')
                state_agent.append(state)
                action_agent.append(action)
                reward_agent.append(reward)
                if done:
                    break
                state = next_state
            state_list.append(state_agent)
            action_list.append(action_agent)
            reward_list.append(reward_agent)
        # print('--------')
        # print(len(state_list))
        # print("Successful random walk")
        self.build_froest(state_list, action_list, reward_list, max_K)
        return state_list, action_list, reward_list, max_K

    def find_dominator(self, graph: np.ndarray, parks: np.ndarray, K: int):
        num_nodes = len(graph)
        prev = -np.ones((len(parks), num_nodes), dtype=np.int32)
        dominators = -np.ones((num_nodes,), dtype=np.int32)
        dominators[parks] = parks
        last_update_list = list(parks)
        for dis in range(1, K+1):
            new_update_list = []
            for node in last_update_list:
                idx_in_parks = np.where(parks == dominators[node])[0].item()
                adj_nodes = np.where(graph[node] != np.inf)[0]
                for adj_node in adj_nodes:
                    if dominators[adj_node] == -1:
                        dominators[adj_node] = dominators[node]
                        new_update_list.append(adj_node)
                        prev[idx_in_parks, adj_node] = node
            last_update_list[:] = new_update_list[:]
        return dominators, prev

    def build_froest(self, state_list, action_list, reward_list, max_K):
        num_nodes = self.u_size
        # 1. build global adjacent matrix and find all parks
        parks = []
        for traj_index in range(len(state_list)):
            reward_ary = np.array(reward_list[traj_index])
            state_ary = np.array(state_list[traj_index])
            action_ary = np.array(action_list[traj_index])
            u_ary = state_ary * self.action_size + action_ary
            parks.extend(list(u_ary[np.where(reward_ary > 0)[0]]))
            
            for step, u in enumerate(u_ary):
                # print(self.u_reward)
                self.u_reward[u] = reward_ary[step]
            traj_length = len(u_ary)

            for idx in range(traj_length-1, 0, -1):
                if u_ary[idx] == u_ary[idx-1]:
                    continue
                self.graph[u_ary[idx], u_ary[idx-1]] =1
                # assert not ((u_ary[idx] == 0) and (u_ary[idx-1] == 3)), "there should be path from 0 to 3 "
            # print("state:")
            # print(state_ary)
            # print("action:")
            # print(action_ary)
            # print("u:")
            # print(u_ary)
            # input("continue...")
            # if 58 in u_ary:
            #     print("u: ")
            #     print(u_ary)
            #     input("continue...")
        parks = np.unique(parks)

        # 2. find the root (dominator) of each node
        dominators, prev_maps = self.find_dominator(self.graph, parks, max_K)
        # print(dominators)

        # prev = self.dijkstra(self.graph, 58)
        # print("| 1 ", end="")
        # pre =  prev[1]
        # while pre >= 0:
        #     print("<- {} ".format(pre), end="")
        #     if pre == prev[pre]:
        #         break
        #     pre = prev[pre]
        # print("")
        # exit()
        # 3. mask the global adjacent matrix to obtain the adajacent matrix of each tree
        print("number of parks: {}".format(len(parks)))
        for idx, park in enumerate(parks):
            self.maps[park] = self.prev_to_map(prev_maps[idx])
            np.savetxt("new_method_pre.txt", prev_maps[idx])
            np.savetxt("maps_version2.txt", self.maps[park], fmt="%.0f")
            self.maps_indicate.append(park)
        print("Build forest successfully!")

class newnewDiscreteGenerateForest:
    def __init__(self, state_size, action_size, batch_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.u_size = self.state_size * self.action_size
        self.batch_size = batch_size
        self.u_reward = torch.from_numpy(np.array([0 for _ in range(self.u_size)])) #每个u获得的奖励
        self.maps = [ None for _ in range(self.u_size)]
        self.pre_maps = [None for _ in range(self.u_size)]
        self.maps_indicate = []
        self.seed = seed
        self.graph = np.zeros((self.u_size, self.u_size)) + np.inf  # the global ajdacent matrix

    def prev_to_map(self, prev_list):
        map = np.ones(shape=[len(prev_list), len(prev_list)])*np.inf
        for u in range(len(prev_list)):
            if prev_list[u] == -1:
                continue
            else:
                map[prev_list[u],u]=1
        return map

    def dijkstra(self, graph, s, dis_threshold=np.inf):
        # print('in dij')
        # print(s)
        # print('-----')
        n = len(graph)
        # 创建一个flag数组，用于保存遍历情况
        flag = np.zeros(shape=(n,), dtype=bool)
        # 创建一个dist数组，用于保存最短路径
        dis = graph[s].copy()
        # print("distance")
        # print(dis)
        # print(dis[s])
        # print("-------")
        # 创建一个prev数组，用于保存对应的最短节点
        prev = np.zeros(shape=(n,), dtype=int) - 1
        prev[np.where(dis<np.inf)[0]] = s
        # 将源节点放入集合S中
        flag[s] = True
        # print(graph)
        # Dijkstra算法中重点：
        # 迭代次数为n-1次，因为如果确定某一节点，但其最小值不会影响其他节点，每次迭代只能确定一个节点；
        # 依次将节点放入集合S中（即已访问过的节点）；
        for i in range(n-1):
            # 找到当前dist中还未被遍历的节点中权值最小的节点；
            # 并将其放入集合S中；
            temp = float('inf')
            u = 0
            for j in range(n):
                if not flag[j] and dis[j] != float('inf') and dis[j] < temp:
                    u = j
                    temp = dis[j]
            if dis[u] > dis_threshold:
                break
            flag[u] = True
            # 确定当前节点最短距离后，其他节点最短距离是否随之改变，若改变，即可确定其最短路径；
            for j in range(n):
                if not flag[j] and graph[u][j] != float('inf'):
                    if dis[u] + graph[u][j] < dis[j] or dis[j] == float('inf'):
                        dis[j] = dis[u] + graph[u][j]
                        prev[j] = u
        # 输出结果
        return prev
    
    def prev_to_map(self, prev_list):
        map = np.ones(shape=[len(prev_list), len(prev_list)])*np.inf
        for u in range(len(prev_list)):
            if prev_list[u] == -1:
                continue
            else:
                map[prev_list[u],u]=1
        return map

    def RandomWalk(self, env, num_agent, num_step, max_K = K):
        state_list = []
        action_list = []
        reward_list  =[]
        # print(num_agent)
        for agent in range(num_agent):
            state_agent = []
            action_agent = []
            reward_agent = []
            state = env.reset()
            for t in range(num_step):
                action = np.random.choice(self.action_size)
                next_state, reward, done, _ = env.step(action)
                # if reward:
                    # print('+++++')
                state_agent.append(state)
                action_agent.append(action)
                reward_agent.append(reward)
                if done:
                    break
                state = next_state
            state_list.append(state_agent)
            action_list.append(action_agent)
            reward_list.append(reward_agent)
        # print('--------')
        # print(len(state_list))
        # print("Successful random walk")
        self.build_froest(state_list, action_list, reward_list, max_K)
        return state_list, action_list, reward_list, max_K

    def find_shortest_paths(self, graph: np.ndarray, parks: np.ndarray, max_K: int):
        num_nodes = len(graph)
        optimized_graph = np.zeros_like(graph) + np.inf
        for park in parks:
            prev_vector_for_this_park = self.dijkstra(self.graph, park, max_K)
            des_nodes = np.where(prev_vector_for_this_park != -1)[0]
            # print(des_nodes)
            # print(prev_vector_for_this_park)
            optimized_graph[prev_vector_for_this_park[des_nodes], des_nodes] = 1
        return optimized_graph

    def build_froest(self, state_list, action_list, reward_list, max_K):
        num_nodes = self.u_size
        # 1. build global adjacent matrix and find all parks
        parks = []
        for traj_index in range(len(state_list)):
            reward_ary = np.array(reward_list[traj_index])
            state_ary = np.array(state_list[traj_index])
            action_ary = np.array(action_list[traj_index])
            u_ary = state_ary * self.action_size + action_ary
            parks.extend(list(u_ary[np.where(reward_ary > 0)[0]]))
            
            for step, u in enumerate(u_ary):
                self.u_reward[u] = reward_ary[step]
            traj_length = len(u_ary)

            for idx in range(traj_length-1, 0, -1):
                if u_ary[idx] == u_ary[idx-1]:
                    continue
                self.graph[u_ary[idx], u_ary[idx-1]] = 1
        parks = np.sort(np.unique(parks))
        print("parks:")
        print(parks)

        # #####################################################################
        # ###                       test data below                         ###
        # ####################################################################
        # self.graph = np.array([[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
        #                        [-1, -1,  1, -1, -1, -1, -1,  1, -1, -1, -1],
        #                        [-1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1],
        #                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1,  1],
        #                        [-1, -1,  1, -1, -1, -1, -1, -1, -1, -1, -1],
        #                        [-1, -1, -1, -1,  1, -1,  1, -1, -1, -1, -1],
        #                        [-1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1],
        #                        [-1, -1, -1, -1, -1, -1, -1, -1,  1, -1, -1],
        #                        [-1, -1, -1, -1, -1, -1, -1, -1, -1,  1, -1],
        #                        [-1, -1, -1,  1, -1, -1, -1, -1, -1, -1, -1],
        #                        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]],
        #                        dtype=np.float32
        #                     )
        # self.graph[np.where(self.graph == -1)] = np.inf
        # parks = np.array([1, 5])
        # #####################################################################
        # ###                       test data over                          ###
        # #####################################################################

        # 2. find the root (dominator) of each node
        optimized_graph = self.find_shortest_paths(self.graph, parks, max_K)
        np.savetxt("self_graph.txt", self.graph, fmt="%.0f")
        np.savetxt("optimized_graph.txt", optimized_graph, fmt="%.0f")
        self.graph[:, :] = optimized_graph[:, :]
        self.maps_indicate.extend(list(parks))

        # for frozen lake only
        # self.maps[58] = self.graph.copy()
        # exit(0)

        # # 3. mask the global adjacent matrix to obtain the adajacent matrix of each tree
        # print("number of parks: {}".format(len(parks)))
        # for idx, park in enumerate(parks):
        #     self.maps[park] = self.prev_to_map(prev_maps[idx])
        #     np.savetxt("new_method_pre.txt", prev_maps[idx])
        #     self.maps_indicate.append(park)


class RawDiscreteGenerateForest:
    def __init__(self, state_size, action_size, batch_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.u_size = self.state_size * self.action_size
        self.batch_size = batch_size
        self.u_reward = torch.from_numpy(np.array([0 for _ in range(self.u_size)])) #每个u获得的奖励
        self.maps = [ None for _ in range(self.u_size)]
        self.pre_maps = [None for _ in range(self.u_size)]
        self.maps_indicate = []
        self.seed = seed
        self.graph = np.zeros((self.u_size, self.u_size)) + np.inf  # the global ajdacent matrix

    def prev_to_map(self, prev_list):
        map = np.ones(shape=[len(prev_list), len(prev_list)])*np.inf
        for u in range(len(prev_list)):
            if prev_list[u] == -1:
                continue
            else:
                map[prev_list[u],u]=1
        return map

    def dijkstra(self, graph, s, dis_threshold=np.inf):
        # print('in dij')
        # print(s)
        # print('-----')
        n = len(graph)
        # 创建一个flag数组，用于保存遍历情况
        flag = np.zeros(shape=(n,), dtype=bool)
        # 创建一个dist数组，用于保存最短路径
        dis = graph[s].copy()
        # print("distance")
        # print(dis)
        # print(dis[s])
        # print("-------")
        # 创建一个prev数组，用于保存对应的最短节点
        prev = np.zeros(shape=(n,), dtype=int) - 1
        prev[np.where(dis<np.inf)[0]] = s
        # 将源节点放入集合S中
        flag[s] = True
        # print(graph)
        # Dijkstra算法中重点：
        # 迭代次数为n-1次，因为如果确定某一节点，但其最小值不会影响其他节点，每次迭代只能确定一个节点；
        # 依次将节点放入集合S中（即已访问过的节点）；
        for i in range(n-1):
            # 找到当前dist中还未被遍历的节点中权值最小的节点；
            # 并将其放入集合S中；
            temp = float('inf')
            u = 0
            for j in range(n):
                if not flag[j] and dis[j] != float('inf') and dis[j] < temp:
                    u = j
                    temp = dis[j]
            if dis[u] > dis_threshold:
                break
            flag[u] = True
            # 确定当前节点最短距离后，其他节点最短距离是否随之改变，若改变，即可确定其最短路径；
            for j in range(n):
                if not flag[j] and graph[u][j] != float('inf'):
                    if dis[u] + graph[u][j] < dis[j] or dis[j] == float('inf'):
                        dis[j] = dis[u] + graph[u][j]
                        prev[j] = u
        # 输出结果
        return prev
    
    def prev_to_map(self, prev_list):
        map = np.ones(shape=[len(prev_list), len(prev_list)])*np.inf
        for u in range(len(prev_list)):
            if prev_list[u] == -1:
                continue
            else:
                map[prev_list[u],u]=1
        return map

    def RandomWalk(self, env, num_agent, num_step, max_K = K):
        state_list = []
        action_list = []
        reward_list  =[]
        for agent in range(num_agent):
            state_agent = []
            action_agent = []
            reward_agent = []
            state = env.reset()
            for t in range(num_step):
                action = np.random.choice(self.action_size)
                next_state, reward, done, _ = env.step(action)
                state_agent.append(state)
                action_agent.append(action)
                reward_agent.append(reward)
                if done:
                    break
                state = next_state
            state_list.append(state_agent)
            action_list.append(action_agent)
            reward_list.append(reward_agent)
        self.build_graph(state_list, action_list, reward_list, max_K)
        return state_list, action_list, reward_list, max_K

    def build_graph(self, state_list, action_list, reward_list, max_K):
        parks = []
        for traj_index in range(len(state_list)):
            reward_ary = np.array(reward_list[traj_index])
            state_ary = np.array(state_list[traj_index])
            action_ary = np.array(action_list[traj_index])
            u_ary = state_ary * self.action_size + action_ary
            parks.extend(list(u_ary[np.where(reward_ary > 0)[0]]))

            for step, u in enumerate(u_ary):
                self.u_reward[u] = reward_ary[step]
            traj_length = len(u_ary)

            for idx in range(traj_length-1, 0, -1):
                if u_ary[idx] == u_ary[idx-1]:
                    continue
                self.graph[u_ary[idx], u_ary[idx-1]] = 1
        parks = np.unique(parks)
        self.maps_indicate.extend(list(parks))
        print("Build the graph successfully!")

    def sample_trajectory_from_graph(self, graph: np.ndarray, parks: np.ndarray, N: int, K: int):
        origins = np.random.choice(parks, size=N, replace=True)
        traj_list = []
        for origin in origins:
            trajectory = [origin]
            now_node = origin
            for step in range(1, K+1):
                # 当前结点的所有邻接结点
                # rtt = np.where(graph[now_node]==1)
                # if len(rtt) == 2:
                #     print(now_node)
                #     print(len(rtt))
                #     print(rtt)
                #     input("check where")
                adjacent_nodes = np.where(graph[now_node]==1)[0]
                # 还没走完k步，就没有邻接结点了，该轨迹为None
                if len(adjacent_nodes) == 0:
                    trajectory = None
                    break
                # 从邻接结点中随机选择一个
                next_node = np.random.choice(adjacent_nodes, size=1).item()
                # print("next node:", next_node.item())

                trajectory.append(next_node)
                # if now_node == next_node:
                    # print(now_node, next_node)
                    # print(adjacent_nodes)
                    # print(graph[now_node])
                    # print(np.where(graph[now_node] == 1)[1])
                    # input("continue...")
                now_node = next_node
            if isinstance(trajectory, list):
                trajectory.reverse()
                traj_list.append(trajectory)
        return traj_list


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.stack([e.state for e in experiences if e is not None])).long().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.stack([e.next_state for e in experiences if e is not None])).long().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, num_root=10, num_branch = 100):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)  # behavior network
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)  # target network
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        self.heat = HeatMap(action_size, num_root, num_branch, BATCH_SIZE, seed)
        # self.tree = DiscreteGenerateForest(self.state_size, self.action_size, BATCH_SIZE, seed)  # DiscreteGenerateForest(self.state_size, self.action_size, BATCH_SIZE, seed)
        # self.new_tree = newDiscreteGenerateForest(self.state_size, self.action_size, BATCH_SIZE, seed)# DiscreteGenerateForest(self.state_size, self.action_size, BATCH_SIZE, seed)
        # self.new_new_tree = newnewDiscreteGenerateForest(self.state_size, self.action_size, BATCH_SIZE, seed)# DiscreteGenerateForest(self.state_size, self.action_size, BATCH_SIZE, seed)
        # self.tree = newnewDiscreteGenerateForest(self.state_size, self.action_size, BATCH_SIZE, seed)
        self.tree = RawDiscreteGenerateForest(self.state_size, self.action_size, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = np.array(state)
        state = torch.from_numpy(state).long().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        ## TODO: compute and minimize the loss
        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        Q_expected = self.qnetwork_local(states).gather(1, actions)  # 固定行号，确认列号

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def step_traj(self, state_traj, action_traj, reward_traj):
        self.heat.add(state_traj, action_traj, reward_traj, K)
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            experiences = self.heat.sample()
            self.learn_by_H(*experiences)
        
    def learn_by_H(self, state_list, action_list, reward_list):
        H = 0.
        if len(state_list) == 0:
            return
        for traj_index in range(len(state_list)):
            state_traj = torch.from_numpy(np.array(state_list[traj_index])).to(device)
            action_traj = torch.from_numpy(np.array(action_list[traj_index])).to(device)
            reward = reward_list[traj_index]
            prob_traj = self.qnetwork_local(state_traj)
            prob_traj = prob_traj[np.arange(len(prob_traj)), action_traj.flatten()]
            traj_H = 0
            p = 1.
            for k in range(len(prob_traj)):
                p *= prob_traj[k]
                traj_H -= p * reward
            H += traj_H
        # Minimize the Hamiltonian Equation
        self.optimizer.zero_grad()
        H.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)   
 
    def step_traj_in_Tree(self, state_traj, action_traj, reward_traj):
        # self.tree.build(state_traj, action_traj, reward_traj, K)
        # Learn every UPDATE_EVERY time steps.
        # self.t_step = (self.t_step + 1) % UPDATE_EVERY
        # if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            # experiences = self.tree.sample()
            # self.learn_by_H_in_Tree(*experiences)
        self.learn_by_H_in_park(K)


    def learn_by_H_in_park(self, K=K):
        H = 0.
        map = self.tree.graph
        parks = self.tree.maps_indicate
        non_reward_u  = torch.where(self.tree.u_reward!=0)[0]
        reward = self.tree.u_reward[non_reward_u]
        non_reward_s = non_reward_u//self.action_size
        non_reward_a = non_reward_u%self.action_size
        # p = self.qnetwork_local(non_reward_s)
        p = self.qnetwork_local(non_reward_s)[torch.arange(len(non_reward_s)),non_reward_a]
        H -= torch.mean(p*reward)
        for k in range(1, K+1):
            sampled_trajectories = self.tree.sample_trajectory_from_graph(map, parks, N=100, K=k)
            num_traj = len(sampled_trajectories)
            if num_traj == 0:
                continue
            reward_list = [0] * num_traj
            prob_list = [1] * num_traj
            for traj_idx in range(num_traj):
                u_tensor = torch.tensor(sampled_trajectories[traj_idx]).long().to(device)
                s_tensor = u_tensor // self.action_size
                prob_tensor = self.qnetwork_local(s_tensor)[np.arange(len(s_tensor)), u_tensor%self.action_size]
                prob_list[traj_idx] = torch.prod(prob_tensor)

                # num_entries_into_the_traj = len(np.where(map[u_tensor[0].item()] == 1)[0])
                # reward_list[traj_idx] = num_entries_into_the_traj * self.tree.u_reward[u_tensor[-1].item()]

                reward_list[traj_idx] = self.tree.u_reward[u_tensor].sum()
                

            probs = torch.stack(prob_list)
            rewards = torch.stack(reward_list)
            # print(self.tree.u_reward)
            # exit()
            H -= torch.mean(probs * rewards)
        print("H: {}\t".format(H), end="")
        self.optimizer.zero_grad()
        H.backward()
        self.optimizer.step()
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

        # for reward_u in self.tree.maps_indicate:
        #     node_prob_list = torch.ones(len(map), requires_grad=False).float().to(device)
        #     node_reward_list = torch.zeros(len(map)).float().to(device)
        #     node_used_list = torch.zeros(len(map)).to(device).detach()
        #     node_list = torch.from_numpy(np.array([reward_u])).long().to(device).detach()
        #     prob = self.qnetwork_local(node_list//self.action_size)[np.arange(len(node_list)),node_list%self.action_size]

        #     mask = torch.zeros_like(node_prob_list, dtype=torch.float)
        #     mask[reward_u] += 1.
        #     delta = node_prob_list * prob[0]
        #     node_prob_list = node_prob_list + mask * delta

        #     node_reward_list[reward_u] = node_reward_list[reward_u] + self.tree.u_reward[reward_u]
        #     node_used_list[reward_u] = 1
        #     for k in range(K):
        #         next_node_list = []
        #         for node in node_list:
        #             sub_nodes = torch.from_numpy(np.where(map[node]==1)[0]).detach()
        #             if len(sub_nodes)==0:
        #                 continue
        #             next_node_list.extend(sub_nodes.tolist())
        #             node_used_list[sub_nodes] = 1
        #             mask = torch.zeros_like(node_prob_list)
        #             mask[sub_nodes] += 1.
        #             delta = node_prob_list * node_prob_list[node] - node_prob_list
        #             node_prob_list = delta * mask + node_prob_list
        #             prob = self.qnetwork_local(sub_nodes//self.action_size)
        #             prob = prob[np.arange(len(prob)), sub_nodes%self.action_size]
        #             node_prob_list[sub_nodes] = node_prob_list[sub_nodes] * prob
        #             node_reward_list[sub_nodes] = node_reward_list[sub_nodes] +  self.tree.u_reward[sub_nodes] + self.tree.u_reward[node]
        #         node_list = np.unique(next_node_list)
        #     H -= torch.sum(node_used_list * node_reward_list * node_prob_list)
        
        # self.optimizer.zero_grad()
        # H.backward()
        # self.optimizer.step()

        # # ------------------- update target network ------------------- #
        # self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)   


 

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
