from spinup.utils.run_utils import ExperimentGrid
from spinup import goaly
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()

    print(args.num_runs)

    eg = ExperimentGrid(name='ppo-bench')
    eg.add('env_name', 'Pendulum-v0', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 200)
    eg.add('steps_per_epoch', 1000)
    eg.add('ac_kwargs:hidden_sizes', [(32, 32)], 'hid')
    eg.add('ac_kwargs:activation', [tf.nn.relu], '')
    eg.add('inverse_kwargs:hidden_sizes', [(32, 32), (64, 64)], 'hid')
    eg.add('inverse_kwargs:activation', [tf.nn.relu], '')
    eg.add('inverse_kwargs:goals_output_activation', [None, tf.nn.sigmoid], '')
    eg.run(goaly, num_cpu=args.cpu)
