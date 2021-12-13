import os
import torch
import numpy as np
import numpy.random as rd
import torch.multiprocessing as mp
from copy import deepcopy
from net import Actor
from net import Critic, ActorSAC
from torch.nn.utils import clip_grad_norm_


def get_episode_return_and_step(env, act, device):
    episode_step = 1
    episode_return = 0.0  # sum of rewards in an episode

    #max_step = env.max_step
    max_step = 1000
    if_discrete =False# env.if_discrete

    state = env.reset()
    for episode_step in range(max_step):
        s_tensor = torch.as_tensor((state,),dtype=torch.float32, device=device).reshape(-1,3)
        a_tensor = act(s_tensor)
        if if_discrete:
            a_tensor = a_tensor.argmax(dim=1)
        action = a_tensor.detach().cpu().numpy()  # not need detach(), because with torch.no_grad() outside
        state, reward, done, _ = env.step(action)
        episode_return += reward
        if done:
            break
    episode_return = getattr(env, 'episode_return', episode_return)
    return episode_return, episode_step

class AgentBase:  # [ElegantRL.2021.11.11]
    def __init__(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
                 learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """initialize

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        :param net_dim: the dimension of networks (the width of neural networks)
        :param state_dim: the dimension of state (the number of state vector)
        :param action_dim: the dimension of action (the number of discrete action)
        :param reward_scale: scale the reward to get a appropriate scale Q value
        :param gamma: the discount factor of Reinforcement Learning

        :param learning_rate: learning rate of optimizer
        :param if_per_or_gae: PER (off-policy) or GAE (on-policy) for sparse reward
        :param env_num: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
        """
        self.gamma = None
        self.states = None
        self.device = None
        self.traj_list = None
        self.action_dim = None
        self.reward_scale = None
        self.if_off_policy = True
        self.env=None
        self.env_num = env_num
        self.explore_rate = 1.0
        self.explore_noise = 0.1
        self.clip_grad_norm = 4.0
        # self.amp_scale = None  # automatic mixed precision
        self.train_iteration = 0
        self.update_epoch = 5
        self.K = 10
        '''attribute'''
        self.explore_env = None
        self.get_obj_critic = None
        self.H_buffer_max_len = 10000000
        self.criterion = torch.nn.SmoothL1Loss()
        self.cri = self.cri_target = self.if_use_cri_target = self.cri_optim = self.ClassCri = None
        self.act = self.act_target = self.if_use_act_target = self.act_optim = self.ClassAct = None
        self.traj_s = [[] for _ in range(self.K)]
        self.traj_r = [[] for _ in range(self.K)]
        self.traj_a = [[] for _ in range(self.K)]
        self.traj_index = [[] for _ in range(self.K)]
        assert isinstance(gpu_id, int)
        assert isinstance(env_num, int)
        assert isinstance(net_dim, int)
        assert isinstance(state_dim, int)
        assert isinstance(action_dim, int)
        assert isinstance(if_per_or_gae, bool)
        assert isinstance(gamma, float)
        assert isinstance(reward_scale, float)
        assert isinstance(learning_rate, float)

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,
             learning_rate=1e-4, if_per_or_gae=False, env_num=1, gpu_id=0):
        """initialize the self.object in `__init__()`

        replace by different DRL algorithms
        explict call self.init() for multiprocessing.

        :param net_dim: the dimension of networks (the width of neural networks)
        :param state_dim: the dimension of state (the number of state vector)
        :param action_dim: the dimension of action (the number of discrete action)
        :param reward_scale: scale the reward to get a appropriate scale Q value
        :param gamma: the discount factor of Reinforcement Learning

        :param learning_rate: learning rate of optimizer
        :param if_per_or_gae: PER (off-policy) or GAE (on-policy) for sparse reward
        :param env_num: the env number of VectorEnv. env_num == 1 means don't use VectorEnv
        :param gpu_id: the gpu_id of the training device. Use CPU when cuda is not available.
        """
        self.gamma = gamma
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        # self.amp_scale = torch.cuda.amp.GradScaler()
        self.traj_list = [list() for _ in range(env_num)]
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")

        self.cri = self.ClassCri(int(net_dim * 1.25), state_dim, action_dim).to(self.device)
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) if self.ClassAct else self.cri
        self.cri_target = deepcopy(self.cri) if self.if_use_cri_target else self.cri
        self.act_target = deepcopy(self.act) if self.if_use_act_target else self.act
        
        

        self.cri_optim = torch.optim.Adam(self.cri.parameters(), learning_rate)
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate) if self.ClassAct else self.cri
        self.act_optim_H = torch.optim.Adam(self.act.parameters(), 0.01 * learning_rate) if self.ClassAct else self.cri
        assert isinstance(if_per_or_gae, bool)
        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env

    def select_action(self, state: np.ndarray) -> np.ndarray:
        s_tensor = torch.as_tensor(state[np.newaxis], device=self.device)
        a_tensor = self.act(s_tensor)
        action = a_tensor.detach().cpu().numpy()
        return action

    def select_actions(self, state: torch.Tensor) -> torch.Tensor:
        """Select continuous actions for exploration

        :param state: states.shape==(batch_size, state_dim, )
        :return: actions.shape==(batch_size, action_dim, ),  -1 < action < +1
        """

        action = self.act(state.to(self.device))
        if rd.rand() < self.explore_rate:  # epsilon-greedy
            action = (action + torch.randn_like(action) * self.explore_noise).clamp(-1, 1)
        return action.detach().cpu()
    
    def explore_env_traj(self, env, num_agent, num_steps):
        traj_s = [[] for _ in range(self.K)]
        traj_a = [[] for _ in range(self.K)]
        traj_r = [[] for _ in range(self.K)]
        self.env = env
        state = self.env.reset()
        last_done = 0
        for num_agent_i in range(num_agent):
            state  = self.env.reset()
            single_traj_s = []
            single_traj_a = []
            single_traj_r = []
            for num_steps_i in range(num_steps):
                # state = torch.as_tensor(state, dtype=torch.float32)
                #action = np.random.choice(self.action_dim) # self.select_actions(state.unsqueeze(0))[0].numpy()
                action = self.env.action_space.sample()
                next_state, reward, done, _ = env.step(action)
                # print(reward)
                # print(state.shape)
                single_traj_s.append(state.tolist())
                single_traj_a.append(action)
                single_traj_r.append(reward)

                # if reward!=0:
                #     for k in range(self.K):
                #         index = len(single_traj_s)-1-k
                #         if index<0:
                #             break
                #         traj_s[k].append([single_traj_s[index:]])
                #         traj_a[k].append([single_traj_a[index:]])
                #         traj_r[k].append([single_traj_r[index:]])
                if done:
                    break
                else:
                    state = next_state
            max_r_index = np.argmax(np.array(single_traj_r))
            # print(max_r_index)
            traj_s[0].append([single_traj_s[max_r_index]])
            traj_a[0].append([single_traj_a[max_r_index]])
            traj_r[0].append([single_traj_r[max_r_index]])
            for k in range(1,self.K):
                index = max_r_index-k
                if index<0:
                    break
                traj_s[k].append([single_traj_s[index:max_r_index+1]])
                traj_a[k].append([single_traj_a[index:max_r_index+1]])
                traj_r[k].append([single_traj_r[index:max_r_index+1]])
        for k in range(self.K):
            if len(traj_s[k])==0:
                break
            self.traj_s[k].extend(traj_s[k])
            self.traj_a[k].extend(traj_a[k])
            self.traj_r[k].extend(traj_r[k])
            self.traj_index[k].append(np.sum(traj_r[k]))
        for k in range(self.K):
            while(len(self.traj_s[k])>self.H_buffer_max_len):
                self.traj_s[k].pop(0)
                self.traj_a[k].pop(0)
                self.traj_r[k].pop(0)
                self.traj_index[k].pop(0)
        return

    def explore_one_env(self, env, target_step: int) -> list:
        """actor explores in single Env, then returns the trajectory (env transitions) for ReplayBuffer

        :param env: RL training environment. env.reset() env.step()
        :param target_step: explored target_step number of step in env
        :return: `[traj_env_0, ]`
        `traj_env_0 = [(state, reward, mask, action, noise), ...]` for on-policy
        `traj_env_0 = [(state, other), ...]` for off-policy
        """
        state = self.states[0]
        traj = list()
        for _ in range(target_step):
            ten_state = torch.as_tensor(state, dtype=torch.float32)
            ten_action = self.select_actions(ten_state.unsqueeze(0))[0]
            action = ten_action.numpy()
            next_s, reward, done, _ = env.step(action)

            ten_other = torch.empty(2 + self.action_dim)
            ten_other[0] = reward
            ten_other[1] = done
            ten_other[2:] = ten_action
            traj.append((ten_state, ten_other))

            state = env.reset() if done else next_s

        self.states[0] = state

        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [(traj_state, traj_other), ]
        return self.convert_trajectory(traj_list)  # [traj_env_0, ]

    def explore_vec_env(self, env, target_step: int) -> list:
        """actor explores in VectorEnv, then returns the trajectory (env transitions) for ReplayBuffer

        :param env: RL training environment. env.reset() env.step(). It should be a vector env.
        :param target_step: explored target_step number of step in env
        :return: `[traj_env_0, ]`
        `traj_env_0 = [(state, reward, mask, action, noise), ...]` for on-policy
        `traj_env_0 = [(state, other), ...]` for off-policy
        """
        ten_states = self.states

        traj = list()
        for _ in range(target_step):
            ten_actions = self.select_actions(ten_states)
            ten_next_states, ten_rewards, ten_dones = env.step(ten_actions)

            ten_others = torch.cat((ten_rewards.unsqueeze(0), ten_dones.unsqueeze(0), ten_actions))
            traj.append((ten_states, ten_others))
            ten_states = ten_next_states

        self.states = ten_states

        # traj = [(env_ten, ...), ...], env_ten = (env1_ten, env2_ten, ...)
        traj_state = torch.stack([item[0] for item in traj])
        traj_other = torch.stack([item[1] for item in traj])
        traj_list = [(traj_state[:, env_i, :], traj_other[:, env_i, :])
                     for env_i in range(len(self.states))]
        # traj_list = [traj_env_0, ...], traj_env_0 = (ten_state, ten_other)
        return self.convert_trajectory(traj_list)  # [traj_env_0, ...]

    def update_net(self, buffer, batch_size: int, repeat_times: float, soft_update_tau: float) -> tuple:
        """update the neural network by sampling batch data from ReplayBuffer

        :param buffer: Experience replay buffer
        :param batch_size: sample batch_size of data for Stochastic Gradient Descent
        :param repeat_times: `batch_sampling_times = int(target_step * repeat_times / batch_size)`
        :param soft_update_tau: soft target update: `target_net = target_net * (1-tau) + current_net * tau`,
        """

    def optim_update(self, optimizer, objective):  # [ElegantRL 2021.11.11]
        """minimize the optimization objective via update the network parameters

        :param optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
        :param objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
        """
        optimizer.zero_grad()
        objective.backward()
        clip_grad_norm_(parameters=optimizer.param_groups[0]['params'],
                        max_norm=self.clip_grad_norm)
        optimizer.step()

    # def optim_update_amp(self, optimizer, objective):  # automatic mixed precision
    #     """minimize the optimization objective via update the network parameters
    #
    #     amp: Automatic Mixed Precision
    #
    #     :param optimizer: `optimizer = torch.optim.SGD(net.parameters(), learning_rate)`
    #     :param objective: `objective = net(...)` the optimization objective, sometimes is a loss function.
    #     :param params: `params = net.parameters()` the network parameters which need to be updated.
    #     """
    #     # self.amp_scale = torch.cuda.amp.GradScaler()
    #
    #     optimizer.zero_grad()
    #     self.amp_scale.scale(objective).backward()  # loss.backward()
    #     self.amp_scale.unscale_(optimizer)  # amp
    #
    #     # from torch.nn.utils import clip_grad_norm_
    #     # clip_grad_norm_(model.parameters(), max_norm=3.0)  # amp, clip_grad_norm_
    #     self.amp_scale.step(optimizer)  # optimizer.step()
    #     self.amp_scale.update()  # optimizer.step()

    @staticmethod
    def soft_update(target_net, current_net, tau):
        """soft update target network via current network

        :param target_net: update target network via current network to make training more stable.
        :param current_net: current network update via an optimizer
        :param tau: tau of soft target update: `target_net = target_net * (1-tau) + current_net * tau`
        """
        for tar, cur in zip(target_net.parameters(), current_net.parameters()):
            tar.data.copy_(cur.data * tau + tar.data * (1.0 - tau))

    def save_or_load_agent(self, cwd: str, if_save: bool):
        """save or load training files for Agent

        :param cwd: Current Working Directory. ElegantRL save training files in CWD.
        :param if_save: True: save files. False: load files.
        """

        def load_torch_file(model_or_optim, _path):
            state_dict = torch.load(_path, map_location=lambda storage, loc: storage)
            model_or_optim.load_state_dict(state_dict)

        name_obj_list = [('actor', self.act), ('act_target', self.act_target), ('act_optim', self.act_optim),
                         ('critic', self.cri), ('cri_target', self.cri_target), ('cri_optim', self.cri_optim), ]
        name_obj_list = [(name, obj) for name, obj in name_obj_list if obj is not None]

        if if_save:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                torch.save(obj.state_dict(), save_path)
        else:
            for name, obj in name_obj_list:
                save_path = f"{cwd}/{name}.pth"
                load_torch_file(obj, save_path) if os.path.isfile(save_path) else None

    def convert_trajectory(self, traj_list: list) -> list:  # off-policy
        """convert trajectory (env exploration type) to trajectory (replay buffer type)

        convert `other = concat((      reward, done, ...))`
        to      `other = concat((scale_reward, mask, ...))`

        :param traj_list: `traj_list = [(tensor_state, other_state), ...]`
        :return: `traj_list = [(tensor_state, other_state), ...]`
        """
        for ten_state, ten_other in traj_list:
            ten_other[:, 0] = ten_other[:, 0] * self.reward_scale  # ten_reward
            ten_other[:, 1] = (1.0 - ten_other[:, 1]) * self.gamma  # ten_mask = (1.0 - ary_done) * gamma
        return traj_list


class AgentDDPG_H(AgentBase):  # [ElegantRL.2021.11.11]
    """Modified SAC
    - reliable_lambda and TTUR (Two Time-scale Update Rule)
    - modified REDQ (Randomized Ensemble Double Q-learning)
    """
    def __init__(self):
        AgentBase.__init__(self)
        self.ClassCri = Critic
        self.get_obj_critic = self.get_obj_critic_raw
        self.ClassAct = ActorSAC
        self.if_use_cri_target = True
        self.if_use_act_target = False
        self.alpha_log = None
        self.alpha_optim = None
        self.target_entropy = None
        self.obj_critic = (-np.log(0.5)) ** 0.5  # for reliable_lambda
    
    

    def init(self, net_dim=256, state_dim=8, action_dim=2, reward_scale=1.0, gamma=0.99,learning_rate=3e-4, if_per_or_gae=False, env_num=1, gpu_id=0, G=20, M=2, N=10,K=10,update_epoch=5,samples=100,H_gamma=0.01):
        #mp.set_start_method('spawn',force=True)
        self.gamma = gamma
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.reward_scale = reward_scale
        self.traj_list = [list() for _ in range(env_num)]
        self.G = G
        self.M = M
        self.N = N
        self.K = K
        self.update_epoch = update_epoch
        self.samples = samples
        self.H_gamma = H_gamma
        self.device = torch.device(f"cuda:{gpu_id}" if (torch.cuda.is_available() and (gpu_id >= 0)) else "cpu")
        self.cri_list = [self.ClassCri(net_dim, state_dim, action_dim).to(self.device) for i in range(self.N)]
        self.act = self.ClassAct(net_dim, state_dim, action_dim).to(self.device) 
        self.cri_target_list = [deepcopy(self.cri_list[i])for i in range(N)]
        self.cri_optim_list = [torch.optim.Adam(self.cri_list[i].parameters(), learning_rate) for i in range(self.N)]
        self.act_optim = torch.optim.Adam(self.act.parameters(), learning_rate)
        self.act_optim_H = torch.optim.Adam(self.act.parameters(), self.H_gamma * learning_rate) if self.ClassAct else self.cri
        self.train_reward = []
        assert isinstance(if_per_or_gae, bool)
        if env_num == 1:
            self.explore_env = self.explore_one_env
        else:
            self.explore_env = self.explore_vec_env
        self.alpha_log = torch.zeros(1,requires_grad=True, device=self.device)  # trainable parameter
        self.alpha_optim = torch.optim.Adam([self.alpha_log], lr=learning_rate)
        self.target_entropy = np.log(action_dim)
        self.criterion = torch.nn.MSELoss()
        self.alpha = self.alpha_log.cpu().exp().item()
        
         
    def get_obj_critic_raw(self, buffer, batch_size, alpha):
        with torch.no_grad():
            batch = buffer.sample_batch(batch_size)
            state  = torch.Tensor(batch['obs1']).to(self.device)
            next_s = torch.Tensor(batch['obs2']).to(self.device)
            action= torch.Tensor(batch['acts']).to(self.device)
            reward = torch.Tensor(batch['rews']).unsqueeze(1).to(self.device)
            mask = torch.Tensor(batch['done']).unsqueeze(1).to(self.device)
            #state, next_s, actions, reward, mask = buffer.sample_batch(batch_size)
            #print(batch_size,reward.shape,mask.shape,action.shape, state.shape, next_s.shape)
            next_a, next_log_prob = self.act.get_action_logprob(next_s)  # stochastic policy
            g = torch.Generator()
            g.manual_seed(torch.randint(high = 10000000,size = (1,))[0].item())
            a = torch.randperm(self.N ,generator = g)
            #a = np.random.choice(self.N, self.M, replace=False)
            #print(a[:M])
            q_tmp = [self.cri_target_list[a[j]](next_s, next_a) for j in range(self.M)]
            q_prediction_next_cat = torch.cat(q_tmp, 1)
            min_q, min_indices = torch.min(q_prediction_next_cat, dim=1, keepdim=True)
            next_q_with_log_prob = min_q - alpha * next_log_prob
            y_q = reward + (1-mask) * self.gamma * next_q_with_log_prob
        q_values = [self.cri_list[j](state, action) for j in range(self.N)] # todo ensemble
        q_values_cat = torch.cat(q_values,dim=1)
        y_q = y_q.expand(-1, self.N) 
        obj_critic = self.criterion(q_values_cat, y_q) * self.N
        return obj_critic, state
        #return y_q, state,action

    def select_actions(self, state,size, env):
        # if size < 5000:
        #    return env.action_space.sample()
        #else:
        state = state.to(self.device)
        actions = self.act.get_action(state)
        return actions.detach().cpu()
    
    

    def cri_multi_train(self, k):
        #state = self.state
        #action = self.action
        q_values = self.cri_list[k](self.state,self.action)
        obj = self.criterion(q_values, self.y_q)
        self.cri_optim_list[k].zero_grad()
        obj.backward()
        self.cri_optim_list[k].step()
    
    def update_net(self, buffer, batch_size, soft_update_tau):
        #buffer.update_now_len()
        for i in range(self.G):
            '''objective of critic (loss function of critic)'''
            obj_critic, state = self.get_obj_critic(buffer, batch_size, self.alpha)
            #self.y_q, self.state,self.action = self.get_obj_critic(buffer, batch_size, alpha) 
            for q_i in range(self.N):
                self.cri_optim_list[q_i].zero_grad()
            obj_critic.backward()
           
            if ((i + 1) % self.G == 0) or i == self.G - 1:
                a_noise_pg, logprob = self.act.get_action_logprob(state)  # policy gradient
                cri_tmp = []
                for j in range(self.N):
                    self.cri_list[j].requires_grad_(False)
                    cri_tmp.append(self.cri_list[j](state, a_noise_pg))
                q_value_pg = torch.cat(cri_tmp, 1)
                q_value_pg = torch.mean(q_value_pg, dim=1, keepdim=True)
                obj_actor = (-q_value_pg + logprob * self.alpha).mean()  # todo ensemble
                self.act_optim.zero_grad()
                obj_actor.backward()
                for j in range(self.N):
                    self.cri_list[j].requires_grad_(True)
                obj_alpha = -(self.alpha_log * (logprob - 1).detach()).mean()
                self.optim_update(self.alpha_optim, obj_alpha)
                self.alpha = self.alpha_log.cpu().exp().item()
                #print(obj_critic, obj_actor, obj_alpha)
       	    
            for q_i in range(self.N):
                self.cri_optim_list[q_i].step()
            if((i + 1) % self.G == 0) or i == self.G - 1:
                self.act_optim.step()
            for q_i in range(self.N):
                self.soft_update(self.cri_target_list[q_i], self.cri_list[q_i], soft_update_tau)
            num_eval = 50
            if (self.train_iteration)%self.update_epoch == 0:
                if self.train_iteration%(10*self.update_epoch)==0:
                    self.explore_env_traj(self.env, 100, 1000)
                # print('hello')
                H = 0.
                flag = False
                for k in range(self.K):
                    if len(self.traj_s[k]) == 0:
                        break
                    if len(self.traj_s[k])>self.samples:
                        index = np.random.choice(len(self.traj_s[k]),replace=False,size=self.samples)
                    else:
                        index = np.arange(len(self.traj_s[k]))
                    flag = True
                    s = np.array(self.traj_s[k])[index]
                    a = np.array(self.traj_a[k])[index]
                    r = np.array(self.traj_r[k])[index]
                    
                    s = s.reshape(len(s),k+1,self.state_dim)
                    r = r.reshape(len(r),k+1)
                    
                    
                    x = np.arange(len(index))
                    s_vec = torch.from_numpy(np.array(s).reshape(-1,self.state_dim)).to(self.device).float()
                    a_vec = torch.from_numpy(np.array(a).reshape(-1,self.action_dim)).to(self.device).long()
                    r = torch.from_numpy(r).to(self.device)
                    
                    p = self.act.get_prob(s_vec,a_vec).reshape(len(index),(k+1)*self.action_dim)
                    traj_p = torch.prod(p, dim=1).squeeze()
                    
                    traj_r = torch.sum(r, dim=1).squeeze()
                    
                    H -= torch.sum(traj_p * traj_r) 

                if flag:
                    self.optim_update(self.act_optim_H, H)

        if self.train_iteration%5==0:
            ep = 0.
            num_eval = 5
            for eval_i in range(num_eval):
                episode_reward, _, = get_episode_return_and_step(self.env, self.act,self.device)
                ep += episode_reward 
                
            self.train_reward.append([ep/num_eval,self.train_iteration])
            print(self.train_iteration,ep/num_eval)
        self.train_iteration += 1
        return obj_actor, self.alpha
