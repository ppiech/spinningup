import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import numpy as np
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph

def load_policy(fpath, itr='last', deterministic=False, goal_octaves=5):

    # handle which epoch to load from
    if itr=='last':
        saves = [int(x[11:]) for x in os.listdir(fpath) if 'simple_save' in x and len(x)>11]
        itr = '%d'%max(saves) if len(saves) > 0 else ''
    else:
        itr = '%d'%itr

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, osp.join(fpath, 'simple_save'+itr))

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    def get_action(prev_goal, x):
        goal = sess.run(model['goals_pi'], feed_dict={model['x']: x[None,:]})
        # goal = np.array([12])
        action = sess.run(action_op, feed_dict={model['x']: x[None,:], model['goals_ph']: goal})[0]
        return goal, action

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars'+itr+'.pkl'))
        env = state['env']
    except e:
        print(e)
        env = None

    return env, get_action

def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    o, r, d, ep_ret, ep_len, n, g = env.reset(), 0, False, 0, 0, 0, [0]
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)

        prev_g = g
        g, a = get_action(prev_g, o)

        if prev_g[0] != g[0]:
            print(' ' + str(g[0]), end='', flush=True)
        else:
            print('.', end='', flush=True)

        o, r, d, _ = env.step(a)

        #reward standing still:
        #r =  -abs(o[0] + 0.5) - abs(o[1])* 10

        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('')
            print('Episode %d \t EpRet %.3f \t EpLen %d'%(n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--goal_octaves', '-o', type=int, default=4)
    parser.add_argument('--deterministic', '-d', action='store_true')
    args = parser.parse_args()
    env, get_action = load_policy(args.fpath,
                                  args.itr if args.itr >=0 else 'last',
                                  args.deterministic,
                                  args.goal_octaves)
    run_policy(env, get_action, args.len, args.episodes, not(args.norender))
