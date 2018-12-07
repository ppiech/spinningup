import numpy as np
import tensorflow as tf
import gym
import time
from spinup.algos.ppo.ppo import ppo
import spinup.algos.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

from spinup.pawel.stability import stability_reward, cluster_traces


def pawel(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):

    episode_observations = []

    def bonus(observation, reward):
        episode_observations[-1].append(observation[0])
        bonus = stability_reward(np.array(episode_observations[-1]))
        return bonus

    def on_reset(observation):
        episode_observations.append([observation[0]])

    def policy_env():
        wrapper = EnvWrapper(env_fn(), bonus, on_reset)
        return wrapper

    ppo(policy_env, actor_critic, ac_kwargs, seed, steps_per_epoch, epochs, gamma, clip_ratio, pi_lr,
        vf_lr, train_pi_iters, train_v_iters, lam, max_ep_len, target_kl, logger_kwargs, save_freq)

class EnvWrapper(gym.Env):
    def __init__(self, env, bonus_function, on_reset_function):
        self.env = env
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

        self.bonus_function = bonus_function
        self.on_reset_function = on_reset_function

    def seed(self, seed=None):
        self.env.seed(seed)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        reward = self.bonus_function(observation, reward)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        self.on_reset_function(observation)
        return observation

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--steps', type=int, default=4000)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='pawel')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    pawel(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
