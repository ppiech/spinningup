from spinup.utils.run_utils import ExperimentGrid
from spinup import goaly
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()

    eg = ExperimentGrid(name='goaly-bench')
    eg.add('env_name', 'CartPole-v0', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 100)
    eg.add('goal_octaves', [6])
    eg.add('inverse_buffer_size', [3])
    eg.add('no_step_reward', [True], '')
    eg.add('steps_per_epoch', 4000)
    eg.add('ac_kwargs:hidden_sizes', [(32,32)], 'hid')
    eg.add('ac_kwargs:activation', [tf.tanh, tf.nn.relu], '')
    eg.run(goaly, num_cpu=args.cpu)
