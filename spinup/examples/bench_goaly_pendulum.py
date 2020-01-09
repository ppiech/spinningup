from spinup.utils.run_utils import ExperimentGrid
from spinup import goaly
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=10)
    args = parser.parse_args()

    print(args.num_runs)

    eg = ExperimentGrid(name='no_stability_in_goal')
    eg.add('env_name', 'Pendulum-v0', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 400)
    eg.add('steps_per_epoch', 2400)
    eg.add('goal_octaves', [6])
    eg.add('no_path_len_reward', [True])
    eg.add('no_step_reward', [True], '')
    eg.add('actions_step_reward', [True], '')
    eg.add('forward_error_for_curiosity_reward', [False], '')
    eg.add('finish_action_path_on_new_goal', [True], '')
    eg.add('ac_kwargs:hidden_sizes', [(32, 32)], 'hid')
    eg.add('ac_kwargs:activation', [tf.nn.relu], '')
    eg.add('inverse_kwargs:hidden_sizes', [(32, 32)], 'hid')
    eg.add('inverse_kwargs:activation', [tf.nn.relu], '')
    eg.add('inverse_kwargs:inverse_buffer_size', [3], '')
    eg.add('inverse_kwargs:goals_output_activation', [tf.nn.sigmoid], '')
    eg.run(goaly, num_cpu=args.cpu)
