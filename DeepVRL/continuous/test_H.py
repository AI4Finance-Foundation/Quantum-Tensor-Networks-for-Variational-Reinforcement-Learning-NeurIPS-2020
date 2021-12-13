from run_H import *
from ddpg_H import  AgentDDPG_H
from env import PreprocessEnv
import torch.multiprocessing as mp
import gym
import numpy as np
import argparse
import traj_env
parser = argparse.ArgumentParser()
parser.add_argument(
        '--K',
        type=int,
        default=10,
    )
parser.add_argument(
        '--gamma',
        type=float,
        default=0.001,
    )
parser.add_argument(
        '--update_epoch',
        type=int,
        default=5,
    )
parser.add_argument(
        '--num_sample',
        type=int,
        default=10,
    )
parser.add_argument(
        '--aid',
        type=int,
        default=0,
    )
main_args = parser.parse_args()

if __name__ == '__main__':
    
    mp.set_start_method("spawn")
    gym.logger.set_level(40) # Block warning    
    #env = mpe_make_env('simple_spread')
    env = gym.make("Hopper-v2") # traj_env.EpisodicRewardsEnv("Hopper-v2")
    args = Arguments(if_off_policy=True)  # AgentSAC(), AgentTD3(), AgentDDPG()
    args.agent = AgentDDPG_H()
    args.env = PreprocessEnv(env)
    args.reward_scale = 2 ** -1  # RewardRange: -200 < -150 < 300 < 334
    args.gamma = 0.95
    args.marl=False
    args.max_step = 100
    args.n_agents = 3
    args.visible_gpu = '3,4,5,6,7'
    args.rollout_num = 2# the number of rollout workers (larger is not always faster)
    args.main_args = main_args
    train_and_evaluate(args) # the training process will terminate once it reaches the target reward.
