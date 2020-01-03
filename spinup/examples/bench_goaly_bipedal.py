from spinup.utils.run_utils import ExperimentGrid
from spinup import goaly
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=10)
    args = parser.parse_args()

    eg = ExperimentGrid(name='discounts')
    eg.add('env_name', 'BipedalWalker-v2', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('goal_octaves', [6])
    eg.add('invese_buffer_size', [3])
    eg.add('goal_discount_rate', [0.0, 0.1])
    eg.add('no_path_len_reward', [True])
    eg.add('forward_error_for_curiosity_reward', [True], '')
    eg.add('epochs', 500)
    eg.add('steps_per_epoch', 1000)
    eg.add('ac_kwargs:hidden_sizes', [(32,32)], 'hid')
    eg.add('ac_kwargs:activation', [tf.nn.relu], '')
    eg.run(goaly, num_cpu=args.cpu)
