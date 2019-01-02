import numpy as np
import tensorflow as tf
import gym
import time
import random
import matplotlib.pyplot as plt

from gym import spaces

from spinup.algos.ppo.ppo import ppo
import spinup.algos.ppo.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs

from spinup.pawel.stability import stability_reward, spans, cluster_traces, is_end_of_span
from spinup.pawel.kshape import distance

DEFAULT_WINDOW_SIZE = 64
DEFAULT_MEMORY_DECAY = 0.9
DEFAULT_NUMBER_OF_GOALS = 2

class PawelBuffer:
    def __init__(self):
        self.step_num = 0
        self.observations = []
        self.spans = []
        self.epoch_num = 0

    def decay_spans(self, decay_rate):
        spans_length = len(self.spans)
        number_to_retain = int(spans_length * decay_rate)
        for i in range(number_to_retain):
            index_to_swap = random.randint(0, spans_length - 1)
            swap = self.spans[i]
            self.spans[i] = self.spans[index_to_swap]
            self.spans[index_to_swap] = swap
        self.spans = self.spans[0:number_to_retain]

class PawelPolicy:
    def __init__(self, num_goals = DEFAULT_NUMBER_OF_GOALS, window_size = DEFAULT_WINDOW_SIZE):
        self.window_size = window_size
        self.num_goals = num_goals
        self.goal_observations = np.identity(num_goals)
        self.centroids = np.zeros((num_goals, window_size))

        self.current_goal = 0

    def current_goal_observation(self):
        return self.goal_observations[self.current_goal]

    def current_centroid(self):
        return self.centroids[self.current_goal]

def pawel(env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(), seed=0,
        steps_per_epoch=4000, epochs=50, gamma=0.99, clip_ratio=0.2, pi_lr=3e-4,
        vf_lr=1e-3, train_pi_iters=80, train_v_iters=80, lam=0.97, max_ep_len=1000,
        target_kl=0.01, logger_kwargs=dict(), save_freq=10):

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    buffer = PawelBuffer()
    policy = PawelPolicy()

    def on_step(observation, reward):
        buffer.step_num += 1
        if buffer.step_num >= steps_per_epoch:
            update()
            buffer.step_num = 0
            buffer.epoch_num += 1
        buffer.observations[-1].append(observation[0])

        stability_bonus = stability_reward(np.array(buffer.observations[-1]))
        logger.store(StabilityReward=stability_bonus)
        end_of_span_bonus = end_of_span_reward(buffer.observations[-1])
        logger.store(EndOfSpanReward=end_of_span_bonus)

        bonus = stability_bonus + end_of_span_bonus
        #bonus = end_of_span_reward(buffer.observations[-1])
        return np.concatenate((observation, policy.current_goal_observation())), bonus

    def on_reset(observation):
        buffer.observations.append([observation[0]])
        return np.concatenate((observation, policy.current_goal_observation()))

    def policy_env():
        wrapper = EnvWrapper(env_fn(), on_step, on_reset, policy.num_goals)
        return wrapper

    def update():
        for obs in buffer.observations:
            plt.plot(obs)
        plt.savefig("/tmp/stability/%s-obs.png"%str(buffer.epoch_num))
        plt.close()

        episode_spans = spans(buffer.observations, window=DEFAULT_WINDOW_SIZE)
        buffer.spans.extend(episode_spans)
        policy.centroids, clusters = cluster_traces(buffer.spans, 2, policy.centroids)
        plot_clusters(policy.centroids, clusters, buffer.epoch_num)
        buffer.decay_spans(DEFAULT_MEMORY_DECAY)
        buffer.observations = [[]]

        logger.log_tabular('CentroidDistance', with_min_and_max=True)
        logger.log_tabular('StabilityReward', average_only=True)
        logger.log_tabular('EndOfSpanReward', average_only=True)

        logger.dump_tabular()

    def plot_clusters(centroids, clusters, epoch_num):
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
        for i in range(len(clusters)):
            cluster = clusters[i]
            color = colors[i%(len(colors))]
            # cluster_line = plt.plot(cluster[0])
            # plt.setp(cluster_line, 'color', color, 'linewidth', 3.0)
            for span_num in cluster:
                span_line = plt.plot(buffer.spans[span_num])
                plt.setp(span_line, 'color', color, 'linewidth', 1.0)
        for i in range(len(centroids)):
            span_line = plt.plot(centroids[i])
            color = colors[i%(len(colors))]
            plt.setp(span_line, 'color', color, 'linewidth', 2.0)

        plt.savefig("/tmp/stability/%s.png"%str(epoch_num))
        plt.close()

    def end_of_span_reward(observations):
        if is_end_of_span(observations):
            span = observations[-policy.window_size:]
            distance_to_centroid = distance(span, policy.current_centroid())
            logger.store(CentroidDistance=distance_to_centroid)
            r =  100/(distance_to_centroid + 0.000001) - 1
            return r
        else:
            return 0

    ppo(policy_env, actor_critic, ac_kwargs, seed, steps_per_epoch, epochs, gamma, clip_ratio, pi_lr,
        vf_lr, train_pi_iters, train_v_iters, lam, max_ep_len, target_kl, logger_kwargs, save_freq)

class EnvWrapper(gym.Env):
    def __init__(self, env, on_step_function, on_reset_function, num_goal_observations):
        self.env = env
        observation_space = self.env.observation_space
        print(type (observation_space.low))
        shape = ((observation_space.shape[0] + num_goal_observations))
        observation_space = spaces.Box(
            low=np.concatenate((observation_space.low, np.zeros([num_goal_observations], dtype=int))),
            high=np.concatenate((observation_space.high, np.ones([num_goal_observations], dtype=int))))

        self.observation_space = observation_space
        self.action_space = self.env.action_space

        self.on_step_function = on_step_function
        self.on_reset_function = on_reset_function

    def seed(self, seed=None):
        self.env.seed(seed)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation, reward = self.on_step_function(observation, reward)
        return observation, reward, done, info

    def reset(self):
        observation = self.env.reset()
        observation = self.on_reset_function(observation)
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
