from spinup.utils.run_utils import ExperimentGrid
from spinup import goaly
import tensorflow as tf

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=4)
    parser.add_argument('--num_runs', type=int, default=10)
    args = parser.parse_args()

    eg = ExperimentGrid(name='v2.1-3-inverse_depth')
    eg.add('env_name', 'MountainCar-v0', '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    eg.add('epochs', 2000)
    eg.add('steps_per_epoch', 1000)
    eg.add('num_goal_bits', 5)

    eg.add('goaly_kwargs:Goals2:log_level', 2, '')
    eg.add('goaly_kwargs:Goals2:finish_action_path_on_new_goal', [True], '')
    eg.add('goaly_kwargs:Goals2:gamma', 0.9, '')
    eg.add('goaly_kwargs:Goals2:reward_stability', False, '')
    eg.add('goaly_kwargs:Goals2:reward_curiosity', True, '')
    eg.add('goaly_kwargs:Goals2:reward_external', True, '')
    eg.add('goaly_kwargs:Goals2:ac_kwargs:hidden_sizes', (32, 32), '')

    eg.add('goaly_kwargs:Goals:log_level', 2, '')
    eg.add('goaly_kwargs:Goals:finish_action_path_on_new_goal', [True], '')
    eg.add('goaly_kwargs:Goals:gamma', 0.9, '')
    eg.add('goaly_kwargs:Goals:reward_stability', True, '')
    eg.add('goaly_kwargs:Goals:reward_curiosity', False, '')
    eg.add('goaly_kwargs:Goals:reward_external', False, '')
    eg.add('goaly_kwargs:Goals:ac_kwargs:hidden_sizes', (32, 32), '')
    eg.add('goaly_kwargs:Goals:inverse_kwargs:hidden_sizes', [(32,), (16,16),(32,32)], 'hid')

    eg.add('goaly_kwargs:Actions:log_level', 2, '')
    eg.add('goaly_kwargs:Actions:finish_action_path_on_new_goal', [True], '')
    eg.add('goaly_kwargs:Actions:gamma', 0.99, '')
    eg.add('goaly_kwargs:Actions:reward_stability', True, '')
    eg.add('goaly_kwargs:Actions:reward_external', False, '')
    eg.add('goaly_kwargs:Actions:ac_kwargs:hidden_sizes', (32, 32), '')
    eg.add('goaly_kwargs:Actions:inverse_buffer_size', 3, '')
    eg.add('goaly_kwargs:Actions:inverse_kwargs:hidden_sizes', [(16,16)] , 'hid')


    eg.run(goaly, num_cpu=args.cpu)
