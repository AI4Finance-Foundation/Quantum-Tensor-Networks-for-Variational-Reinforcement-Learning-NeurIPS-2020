import os
import time
import shutil

import torch
import numpy as np
import numpy.random as rd
#import multiprocessing as mp

from env import build_env
from replay import ReplayBuffer, ReplayBufferMP, ReplayBufferMARL, ReplayBufferOn
from evaluator import Evaluator
from tqdm import tqdm
import gym
"""[ElegantRL.2021.09.09](https://github.com/AI4Finance-LLC/ElegantRL)"""


class Arguments:
    def __init__(self, if_off_policy=True):
        self.env = None  # the environment for training
        self.agent = None  # Deep Reinforcement Learning algorithm

        '''Arguments for training'''
        self.gamma = 0.99  # discount factor of future rewards
        self.reward_scale = 2 ** 0  # an approximate target reward usually be closed to 256
        self.learning_rate = 2 ** -15  # 2 ** -14 ~= 3e-5
        self.soft_update_tau = 2 ** -8  # 2 ** -8 ~= 5e-3

        self.if_off_policy = if_off_policy
        if self.if_off_policy:  # (off-policy)
            self.net_dim = 2 ** 8  # the network width
            self.batch_size = self.net_dim  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 0  # repeatedly update network to keep critic's loss small
            self.target_step = 2 ** 10  # collect target_step, then update network
            self.max_memo = 2 ** 21  # capacity of replay buffer
            self.if_per_or_gae = False  # PER for off-policy sparse reward: Prioritized Experience Replay.
        else:
            self.net_dim = 2 ** 9  # the network width
            self.batch_size = self.net_dim * 2  # num of transitions sampled from replay buffer.
            self.repeat_times = 2 ** 3  # collect target_step, then update network
            self.target_step = 2 ** 12  # repeatedly update network to keep critic's loss small
            self.max_memo = self.target_step  # capacity of replay buffer
            self.if_per_or_gae = False  # GAE for on-policy sparse reward: Generalized Advantage Estimation.

        '''Arguments for device'''
        self.env_num = 1  # The Environment number for each worker. env_num == 1 means don't use VecEnv.
        self.worker_num = 2  # rollout workers number pre GPU (adjust it to get high GPU usage)
        self.thread_num = 8  # cpu_num for evaluate model, torch.set_num_threads(self.num_threads)
        self.visible_gpu = '0'  # for example: os.environ['CUDA_VISIBLE_DEVICES'] = '0, 2,'
        self.random_seed = 0  # initialize random seed in self.init_before_training()
        
        '''Arguments for evaluate and save'''
        self.cwd = None  # current work directory. None means set automatically
        self.if_remove = True  # remove the cwd folder? (True, False, None:ask me)
        self.break_step = 2 ** 20  # break training after 'total_step > break_step'
        self.if_allow_break = True  # allow break training when reach goal (early termination)

        self.eval_env = None  # the environment for evaluating. None means set automatically.
        self.eval_gap = 2 ** 7  # evaluate the agent per eval_gap seconds
        self.eval_times1 = 2 ** 3  # number of times that get episode return in first
        self.eval_times2 = 2 ** 4  # number of times that get episode return in second
        self.eval_device_id = -1  # -1 means use cpu, >=0 means use GPU

    def init_before_training(self, if_main):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.set_num_threads(self.thread_num)
        torch.set_default_dtype(torch.float32)

        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.visible_gpu)

        '''env'''
        if self.env is None:
            raise RuntimeError(f'\n| Why env=None? For example:'
                               f'\n| args.env = XxxEnv()'
                               f'\n| args.env = str(env_name)'
                               f'\n| args.env = build_env(env_name), from elegantrl.env import build_env')
        if not (isinstance(self.env, str) or hasattr(self.env, 'env_name')):
            raise RuntimeError('\n| What is env.env_name? use env=PreprocessEnv(env).')

        '''agent'''
        if self.agent is None:
            raise RuntimeError(f'\n| Why agent=None? Assignment `args.agent = AgentXXX` please.')
        if not hasattr(self.agent, 'init'):
            raise RuntimeError(f"\n| why hasattr(self.agent, 'init') == False"
                               f'\n| Should be `agent=AgentXXX()` instead of `agent=AgentXXX`.')
        

        '''cwd'''
        if self.cwd is None:
            agent_name = self.agent.__class__.__name__
            env_name = getattr(self.env, 'env_name', self.env)
            self.cwd = f'./{agent_name}_{env_name}_{self.visible_gpu}'
        if if_main:
            # remove history according to bool(if_remove)
            if self.if_remove is None:
                self.if_remove = bool(input(f"| PRESS 'y' to REMOVE: {self.cwd}? ") == 'y')
            elif self.if_remove:
                shutil.rmtree(self.cwd, ignore_errors=True)
                print(f"| Remove cwd: {self.cwd}")
            os.makedirs(self.cwd, exist_ok=True)


'''single processing training'''

def mpe_make_env(scenario_name, benchmark=False):
    '''
    Creates a MultiAgentEnv object as env. This can be used similar to a gym
    environment by calling env.reset() and env.step().
    Use env.render() to view the environment on the screen.

    Input:
        scenario_name   :   name of the scenario from ./scenarios/ to be Returns
                            (without the .py extension)
        benchmark       :   whether you want to produce benchmarking data
                            (usually only done during evaluation)

    Some useful env properties (see environment.py):
        .observation_space  :   Returns the observation space for each agent
        .action_space       :   Returns the action space for each agent
        .n                  :   Returns the number of Agents
    '''
    from multiagent.environment import MultiAgentEnv
    import multiagent.scenarios as scenarios

    # load scenario from script
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # create world
    world = scenario.make_world()
    # create multiagent environment
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env

def train_and_evaluate(args, agent_id=0):
    H_args = args.main_args
    args.init_before_training(if_main=True)
    env = args.env
    # env = build_env(args.env, if_print=False)
    '''init: Agent'''
    agent = args.agent
    if args.marl:
        agent.init(args.net_dim, env.state_dim, env.action_dim, args.learning_rate,args.marl, args.n_agents, args.if_per_or_gae, args.env_num)
    else:
        agent.init(256, env.state_dim, env.action_dim, reward_scale=1.0, gamma=0.99,
             learning_rate=3e-4, if_per_or_gae=False, env_num=1, gpu_id=0,G=10,M=2,N=10,K=H_args.K,update_epoch=H_args.update_epoch,samples=H_args.num_sample,H_gamma=H_args.gamma)
    #agent.save_or_load_agent(args.cwd, if_save=False)
    '''init Evaluator'''
    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1")
    #eval_env = build_env(env) if args.eval_env is None else args.eval_env
    evaluator = Evaluator(args.cwd, agent_id,  agent.device, eval_env,
                          args.eval_gap, args.eval_times1, args.eval_times2)
    evaluator.save_or_load_recoder(if_save=False)
    '''init ReplayBuffer'''
    agent.if_off_policy = False
    if agent.if_off_policy:
        buffer = ReplayBuffer(max_len=args.max_memo, state_dim=env.state_dim,
                              action_dim= env.action_dim,
                              if_use_per=args.if_per_or_gae)
        #buffer = ReplayBufferMARL(max_len=args.max_memo, state_dim=env.state_dim,
        #                      action_dim= env.action_dim,n_agents = 3,
        #                      if_use_per=args.if_per_or_gae)
        buffer.save_or_load_history(args.cwd, if_save=False)
    else:
        buffer = ReplayBufferOn(agent.state_dim, agent.action_dim, size=int(1e6))

    """start training"""
    cwd = args.cwd
    gamma = args.gamma
    break_step = args.break_step
    batch_size = args.batch_size
    target_step = args.target_step
    repeat_times = args.repeat_times
    reward_scale = args.reward_scale
    if_allow_break = args.if_allow_break
    soft_update_tau = args.soft_update_tau
    del args 
    def seed_all(epoch):
        seed = 0
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        eval_env_seed = (seed + 10000 + seed_shift) % mod_value
        #bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.seed(env_seed)
        env.action_space.np_random.seed(env_seed)
        eval_env.seed(eval_env_seed)
        eval_env.action_space.np_random.seed(eval_env_seed)
        #bias_eval_env.seed(bias_eval_env_seed)
        #bias_eval_env.action_space.np_random.seed(bias_eval_env_seed)
    seed_all(epoch=0)
    agent.if_off_policy = False
    '''choose update_buffer()'''
    agent.env = gym.make("Pendulum-v1")
    if agent.if_off_policy:
        assert isinstance(buffer, ReplayBuffer)

        def update_buffer(_trajectory_list):
            assert 0
            _steps = 0
            _r_exp = 0
            #print(_trajectory_list.shape)
            for _trajectory in _trajectory_list:
                i = 0
                _trajectory = _trajectory
                ten_state = torch.as_tensor([item[0] for item in _trajectory], dtype=torch.float32)
                
                ten_reward = torch.as_tensor([item[1] for item in _trajectory])
                ten_done = torch.as_tensor([item[2] for item in _trajectory])
                ten_action = torch.cat([item[3] for item in _trajectory])
                ten_reward = ten_reward * reward_scale  # ten_reward
                ten_mask = (1.0 - ten_done) * gamma  # ten_mask = (1.0 - ary_done) * gamma
                buffer.extend_buffer(ten_state, torch.cat((ten_reward, ten_mask, ten_action[0]),0))
                #print(ten_state, ten_reward, )
                _steps += ten_state.shape[0]
                _r_exp += ten_reward.mean()  # other = (reward, mask, action)
            return _steps, _r_exp
    else:
        assert isinstance(buffer, ReplayBufferOn)

        def update_buffer(_trajectory):
            print()
            _trajectory = list(map(list, zip(*_trajectory)))  # 2D-list transpose
            print(_trajectory[i] for i in range(len(_trajectory)))
            ten_state = torch.as_tensor(_trajectory[0])
            ten_reward = torch.as_tensor(_trajectory[1], dtype=torch.float32) * reward_scale
            ten_mask = (1.0 - torch.as_tensor(_trajectory[2], dtype=torch.float32)) * gamma  # _trajectory[2] = done
            ten_action = torch.as_tensor(_trajectory[3])
            #ten_noise = torch.as_tensor(_trajectory[4], dtype=torch.float32)

            buffer[:] = (ten_state, ten_action, ten_noise, ten_reward, ten_mask)

            _steps = ten_reward.shape[0]
            _r_exp = ten_reward.mean()
            return _steps, _r_exp

    '''init ReplayBuffer after training start'''
    agent.states = env.reset()
    agent.if_off_policy = False
    if agent.if_off_policy:
        if_load = 0
        if not if_load:
            trajectory = explore_before_training(env, target_step)
            trajectory = [trajectory, ]
            steps, r_exp = update_buffer(trajectory)
            evaluator.total_step += steps

    '''start training loop'''
    if_train = True
    #cnt_train = 0
    state = env.reset()
    epoch = 1
    for cnt_train in tqdm(range(125000)):
        #    while if_train or cnt_train < 2000000:
        
        with torch.no_grad():
            if cnt_train >= 5000:
                actions = agent.select_actions(torch.as_tensor(state, dtype=torch.float32).unsqueeze(0), buffer.size, env)
            else:
                actions = env.action_space.sample()
            next_s, reward, done, _ = env.step(actions)
            done  = False if (cnt_train) % 1000 == 0 else done
            buffer.store(state, actions, reward, next_s, done)
            if done or ((cnt_train) % 1000 == 0):
                state = env.reset()
                seed_all(epoch)
                epoch += 1
            else:
                state = next_s
        if cnt_train > 5000:
            agent.update_net(buffer, batch_size, soft_update_tau)
        if cnt_train % 1000 == 0:
            with torch.no_grad():
                log_t = ()
                temp = evaluator.evaluate_and_save(agent.act, 1000, 0, log_t)
                if_reach_goal, if_save = temp
                if_train = not ((if_allow_break and if_reach_goal)
                                or evaluator.total_step > break_step
                                or os.path.exists(f'{cwd}/stop'))
        
            with open('./DDPG_H/DDPG_H_{}_{}_{}_{}_{}.pkl'.format(H_args.K,H_args.gamma, H_args.aid,H_args.update_epoch,H_args.num_sample),'wb') as file:
                import pickle as pkl
                pkl.dump(agent.train_reward, file)
        agent.train_iteration = cnt_train
    with open('./DDPG_H/DDPG_H_{}_{}_{}_{}_{}.pkl'.format(H_args.K,H_args.gamma, H_args.aid,H_args.update_epoch,H_args.num_sample),'wb') as file:
        import pickle as pkl
        pkl.dump(agent.train_reward, file)
    print(f'| UsedTime: {time.time() - evaluator.start_time:>7.0f} | SavedDir: {cwd}')

    env.close()
    agent.save_or_load_agent(cwd, if_save=True)
    buffer.save_or_load_history(cwd, if_save=True) if agent.if_off_policy else None
    evaluator.save_or_load_recoder(if_save=True)


def explore_before_training(env, target_step):  # for off-policy only
    trajectory = list()

    if_discrete = env.if_discrete
    action_dim = env.action_dim
    state = env.reset()
    step = 0
    k = 0
    while True:
        k = k +1
        if if_discrete:
            action = [rd.randn(action_dim),rd.randn(action_dim),rd.randn(action_dim)]  # assert isinstance(action_int)
            next_s, reward, done, _ = env.step(action)
            other = (reward, done, action)
        else:
            action = rd.uniform(-1, 1, size=action_dim)
            next_s, reward, done, _ = env.step(action)
            other = (reward, done, *action)
        trajectory.append((state, reward, done, action))
        
        if k > 100:
            state = env.reset()
            k = 0
        else:
            state = next_s

        step += 1
        if done and step > target_step:
            break
    return trajectory
