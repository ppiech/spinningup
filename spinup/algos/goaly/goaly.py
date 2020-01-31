import numpy as np
import tensorflow as tf
import gym
import time
import spinup.algos.goaly.core as core
from spinup.utils.logx import Logger, EpochLogger
from spinup.utils.mpi_tf import MpiAdamOptimizer, sync_all_params
from spinup.utils.mpi_tools import mpi_fork, mpi_avg, proc_id, mpi_statistics_scalar, num_procs, broadcast, allgather
from gym.spaces import Box, Discrete

from sklearn.decomposition import PCA


class PPOBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, size, gamma=0.99, lam=0.95):
        self.adv_buf = np.zeros(size, dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.step_rew_buf = np.zeros(size, dtype=np.float32)
        self.ret_buf = np.zeros(size, dtype=np.float32)
        self.val_buf = np.zeros(size, dtype=np.float32)
        self.y_logp_buf = np.zeros(size, dtype=np.float32)
        self.gamma, self.lam = gamma, lam
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, reward, step_reward, valule, y_logp):

        """
        Append one timestep of agent-environment interaction to the buffer.

        step_reward are rewards that are not to be accumulated over the while episode
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.rew_buf[self.ptr] = reward
        self.step_rew_buf[self.ptr] = step_reward
        self.val_buf[self.ptr] = valule
        self.y_logp_buf[self.ptr] = y_logp
        self.ptr += 1

    def path_len(self):
        return self.ptr - self.path_start_idx

    def finish_path(self, last_val=0):
        """
        Call this at the end of a trajectory, or when one gets cut off
        by an epoch ending. This looks back in the buffer to where the
        trajectory started, and uses rewards and value estimates from
        the whole trajectory to compute advantage estimates with GAE-Lambda,
        as well as compute the rewards-to-go for each state, to use as
        the targets for the value function.

        The "last_val" argument should be 0 if the trajectory ended
        because the agent reached a terminal state (died), and otherwise
        should be V(s_T), the value function estimated for the last state.
        This allows us to bootstrap the reward-to-go calculation to account
        for timesteps beyond the arbitrary episode horizon (or epoch cutoff).

        pp: this is inconsistent with PPO implementation, where last val for
        done is the last reward
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        rews = np.append(self.rew_buf[path_slice], 0)
        step_rews = self.step_rew_buf[path_slice]
        vals = np.append(self.val_buf[path_slice], last_val)

        # the next two lines implement GAE-Lambda advantage calculation
        deltas = rews[:-1] + self.gamma * vals[1:] - vals[:-1]
        self.adv_buf[path_slice] = core.discount_cumsum(deltas, self.gamma * self.lam) + step_rews

        # the next line computes rewards-to-go, to be targets for the value function
        rew_cumsum = core.discount_cumsum(rews, self.gamma)[:-1]
        rets = rew_cumsum + step_rews
        self.ret_buf[path_slice] = rets

        # Add non-accumulating step-rewards to returns
        self.path_start_idx = self.ptr

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """
        assert self.ptr == self.max_size    # buffer has to be full before you can get
        self.ptr, self.path_start_idx = 0, 0
        # the next two lines implement the advantage normalization trick
        adv_mean, adv_std = mpi_statistics_scalar(self.adv_buf)
        self.adv_buf = (self.adv_buf - adv_mean) / adv_std
        return [self.adv_buf, self.ret_buf, self.y_logp_buf], adv_mean

class ObservationsActionsAndGoalsBuffer:
    """
    A buffer for storing trajectories experienced by a PPO agent interacting
    with the environment, and using Generalized Advantage Estimation (GAE-Lambda)
    for calculating the advantages of state-action pairs.
    """

    def __init__(self, num_goals, obs_dim, act_dim, size):
        self.num_goals = num_goals
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.new_obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.goals_buf = np.zeros(size , dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.ptr, self.max_size = 0, size

    def store(self, obs, new_obs, goal, act):
        """
        Append one timestep of agent-environment interaction to the buffer.
        """
        assert self.ptr < self.max_size     # buffer has to have room so you can store
        self.obs_buf[self.ptr] = obs
        self.new_obs_buf[self.ptr] = new_obs
        self.goals_buf[self.ptr] = goal
        self.act_buf[self.ptr] = act
        self.ptr += 1

    def get(self):
        """
        Call this at the end of an epoch to get all of the data from
        the buffer, with advantages appropriately normalized (shifted to have
        mean zero and std one). Also, resets some pointers in the buffer.
        """

        if self.ptr == self.max_size:
            to_return = [self.obs_buf, self.new_obs_buf, self.goals_buf, self.act_buf]
        else:
            to_return = [self.obs_buf[:self.ptr], self.new_obs_buf[:self.ptr], self.goals_buf[:self.ptr],
                         self.act_buf[:self.ptr]]

        return to_return

    def reset(self):
        self.ptr = 0

    def all_get(self):
        obs_buf = allgather(self.obs_buf)
        new_obs_buf = allgather(self.new_obs_buf)
        goals_buf = allgather(self.goals_buf)
        act_buf = allgather(self.act_buf)

        to_return = [obs_buf, new_obs_buf, goals_buf, act_buf]

        return to_return

    def is_full(self):
        return self.ptr == self.max_size

    def append(self, buffer):
        """
        Add all entries from given buffer to this one.  If this buffer is full, replace random indeces in current
        buffer, with the new data.
        """

        new_obs, new_new_obs, new_goals, new_act = buffer.all_get()
        buf_len = buffer.ptr

        for mpi_proc_num in range(len(new_obs)):
            for i in range(buf_len):
                if self.ptr == self.max_size:
                    section = self.max_size / self.num_goals
                    goal_num = new_goals[mpi_proc_num][i]
                    insert_at = int(np.random.normal(section * goal_num, section)) % self.max_size
                else:
                    insert_at = self.ptr
                    self.ptr += 1

                self.obs_buf[insert_at] = new_obs[mpi_proc_num][i]
                self.new_obs_buf[insert_at] = new_new_obs[mpi_proc_num][i]
                self.goals_buf[insert_at] = new_goals[mpi_proc_num][i]
                self.act_buf[insert_at] = new_act[mpi_proc_num][i]

class GoalyPolicy:

    def __init__(self, name, logger, logger_kwargs, trace_freq, action_space, observation_space,
                x_ph, x_next_ph, actions_ph, num_goals, actor_critic, steps_per_epoch,
                inverse_kwargs=dict(), ac_kwargs=dict(),
                log_level=1,
                gamma=0.99, lam=0.97, clip_ratio=0.2, pi_lr=3e-4, vf_lr=1e-3,
                reward_stability=True, reward_external=True, finish_action_path_on_new_goal=True,
                train_pi_iters=80, train_value_iters=80, target_kl=0.01,
                inverse_buffer_size=3, inverse_lr=1e-3, train_inverse_iters=20,
                reward_curiosity=False):

        self.name = name
        self.logger = logger
        self.trace_freq = trace_freq
        self.action_space = action_space
        self.observation_space = observation_space
        self.log_level = log_level
        self.x_ph, self.x_next_ph, self.actions_ph = x_ph, x_next_ph, actions_ph
        self.prev_goal = None
        self.train_pi_iters = train_pi_iters
        self.train_value_iters = train_value_iters
        self.target_kl = target_kl
        self.train_inverse_iters = train_inverse_iters
        self.reward_stability = reward_stability
        self.reward_external = reward_external
        self.finish_action_path_on_new_goal = finish_action_path_on_new_goal
        self.reward_curiosity = reward_curiosity

        self.traces_logger = Logger(output_fname="traces-{}.txt".format(name), **logger_kwargs)

        with tf.variable_scope(self.name):
            self.init_models(steps_per_epoch, num_goals, actor_critic, gamma, lam, pi_lr, vf_lr, ac_kwargs, inverse_buffer_size,
                             clip_ratio, inverse_kwargs, inverse_lr)

    def init_models(self, steps_per_epoch, num_goals, actor_critic, gamma, lam, pi_lr, vf_lr, ac_kwargs, inverse_buffer_size,
                    clip_ratio, inverse_kwargs, inverse_lr):
        self.adv_ph = core.placeholder(None, 'adv_ph')
        self.ret_ph = core.placeholder(None, 'ret_ph')
        self.logp_old_ph = core.placeholder(None, 'logp_old_ph')
        self.goals_ph = tf.placeholder(dtype=tf.int32, shape=(None, ), name='goals_ph')

        self.goals_pi_actions_input = tf.one_hot(self.goals_ph, num_goals)
        self.pi, self.logp, self.logp_pi, self.value = actor_critic(
            self.x_ph, self.goals_pi_actions_input, self.actions_ph, action_space=self.action_space, **ac_kwargs)

        obs_dim = self.observation_space.shape
        act_dim = self.action_space.shape

        local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.ppo_buf = PPOBuffer(local_steps_per_epoch, gamma, lam)
        self.trajectory_buf = ObservationsActionsAndGoalsBuffer(num_goals, obs_dim, act_dim, local_steps_per_epoch)
        self.inverse_buf = ObservationsActionsAndGoalsBuffer(num_goals, obs_dim, act_dim, steps_per_epoch*inverse_buffer_size)

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['actions_pi', 'actions_v'])
        self.logger.log('\nNumber of parameters: \t actions_pi: %d, \t actions_v: %d\n'%var_counts)

        self.pi_loss, self.value_loss, self.approx_kl, self.approx_ent, self.clipfrac = core.ppo_objectives(
            self.adv_ph, self.value, self.ret_ph, self.logp, self.logp_old_ph, clip_ratio)

        # Inverse Dynamics Model
        a_inverse, goals_one_hot, a_predicted, goals_predicted_logits, self.goals_predicted = \
            core.inverse_model(self.action_space, self.x_ph, self.x_next_ph, self.actions_ph, self.goals_ph, num_goals, **inverse_kwargs)
        a_range = core.space_range(self.action_space)

        a_as_float = tf.cast(a_inverse, tf.float32)

        inverse_action_diff = tf.abs((a_as_float - a_predicted) / a_range, name='inverse_action_diff')
        self.inverse_action_loss = tf.reduce_mean((a_as_float - a_predicted)**2, name='inverse_action_loss') # For training inverse model

        # if isinstance(env.action_space, Discrete):
            # inverse_action_diff = tf.reshape(tf.reduce_mean(inverse_action_diff, axis=1), [-1, 1], 'inverse_action_diff')
        self.inverse_goal_loss = tf.reduce_mean((tf.cast(goals_one_hot - goals_predicted_logits, tf.float32)**2), name='inverse_goal_loss')
        self.inverse_loss = self.inverse_goal_loss + self.inverse_action_loss

        # Errors used for calculating return after each step.
        # Action error needs to be normalized wrt action amplitude, otherwise the error will drive the model behavior
        # towards small amplitude actions.  This doesn't apply to discrete action spaces, where a_predicted is a set of
        # logits compared againt a_inverse that is a one-hot vector.
        if isinstance(self.action_space, Discrete):
            inverse_action_error_denominator = 1.0
        else:
            inverse_action_error_denominator = tf.math.maximum(((tf.abs(a_as_float + a_predicted)) * 10 / a_range), 1e-4)

        self.inverse_action_error = tf.reduce_mean(inverse_action_diff / inverse_action_error_denominator)

        # when calculating goal error for stability reward, compare numerical goal value
        self.inverse_goal_error = tf.reduce_mean(tf.abs(tf.cast(self.goals_predicted - self.goals_ph, tf.float32)) * 2.0 / num_goals)

        # Forward model
        is_action_space_discrete = isinstance(self.action_space, Discrete)
        self.x_pred = core.forward_model(self.x_ph, self.actions_ph, self.x_next_ph, is_action_space_discrete)

        forward_diff = tf.abs((tf.cast(self.x_ph, tf.float32) - self.x_pred), name='forward_diff')
        self.forward_error = tf.reduce_mean(forward_diff, name='forward_error')
        self.forward_loss = tf.reduce_mean((self.forward_error)**2, name='forward_loss')

        # by default use invese action error in stability reward
        self.stability_reward = 1 + 2*self.inverse_action_error * self.inverse_goal_error - self.inverse_action_error - self.inverse_goal_error

        # cap stability reward
        self.stability_reward =  tf.math.maximum(tf.math.minimum(self.stability_reward, 1.0), -1.0)

        self.train_pi = MpiAdamOptimizer(learning_rate=pi_lr).minimize(self.pi_loss)
        self.train_value = MpiAdamOptimizer(learning_rate=vf_lr).minimize(self.value_loss)

        self.train_inverse = MpiAdamOptimizer(learning_rate=inverse_lr).minimize(self.inverse_loss)
        self.train_forward = MpiAdamOptimizer(learning_rate=inverse_lr).minimize(self.forward_loss)

    def get(self, sess, observations, goal):
        actions, value_t, logp_t = \
            sess.run([self.pi, self.value, self.logp_pi],
                     feed_dict={self.x_ph: observations.reshape(1,-1), self.goals_ph: np.array([goal]) })

        actions, value_t, logp_t = actions[0], value_t[0], logp_t[0]

        store_lam = lambda epoch, episode, reward, new_observations: \
            self.store(sess, epoch, episode, observations, new_observations, actions, goal, reward, value_t, logp_t)
        return actions, store_lam

    def calculate_stability(self, sess, observations, new_observations, actions, goal):
        stability, action_error, goal_error, goals_predicted_t = \
            sess.run([self.stability_reward, self.inverse_action_error, self.inverse_goal_error, self.goals_predicted],
                      feed_dict={self.x_ph: np.array([observations]),
                                 self.x_next_ph: np.array([new_observations]),
                                 self.actions_ph: np.array([actions]),
                                 self.goals_ph: np.array([goal])})

        self.log('StabilityReward', stability)
        self.log('ActionError', action_error)
        self.log('GoalError', goal_error)

        return stability, action_error, goal_error, int(goals_predicted_t[0])

    def calculate_curiosity(self, sess, observations, new_observations, actions):
        forward_prediction_error = sess.run([self.forward_error],
                                            feed_dict={self.x_ph: np.array([observations]),
                                                       self.x_next_ph: np.array([new_observations]),
                                                       self.actions_ph: np.array([actions])})

        self.log('Curiosity', forward_prediction_error)

        return forward_prediction_error[0]

    def log(self, key, value):
        self.logger.storeOne("{} - {}".format(key, self.name), value)

    def log_tabular(self, key, value=None, with_min_and_max=False, average_only=False):
        self.logger.log_tabular("{} - {}".format(key, self.name), value, with_min_and_max, average_only)

    def store(self, sess, epoch, episode, observations, new_observations, actions, goal, reward, v_t, logp_t):

        self.trajectory_buf.store(observations, new_observations, goal, actions)

        external = reward if self.reward_external else 0

        if self.reward_curiosity:
            curiosity = self.calculate_curiosity(sess, observations, new_observations, actions)
        else:
            curiosity = 0

        if self.reward_stability:
            stability, action_error, goal_error, goals_predicted_t = \
                self.calculate_stability(sess, observations, new_observations, actions, goal)
        else:
            stability = 0

        self.ppo_buf.store(external + curiosity, stability, v_t, logp_t)

        self.log('VVals', v_t)
        self.log('PathLen', self.ppo_buf.path_len())

        if episode % self.trace_freq == 0:
            self.traces_logger.log_tabular('Epoch', epoch)
            self.traces_logger.log_tabular('Episode', episode)
            self.traces_logger.log_tabular('Reward', reward)
            self.traces_logger.log_tabular('VVals', v_t)

            for i in range(0, len(observations)):
                self.traces_logger.log_tabular('Observations{}'.format(i), observations[i])
            if isinstance(self.action_space, Discrete):
                self.traces_logger.log_tabular('Actions0'.format(i), actions)
            else:
                for i in range(0, len(actions)):
                    self.traces_logger.log_tabular('Actions{}'.format(i), actions[i])

            self.traces_logger.log_tabular('Goal', goal)
            if self.reward_stability:
                self.traces_logger.log_tabular('GoalPredicted', goals_predicted_t)
                self.traces_logger.log_tabular('Stability', stability)
                self.traces_logger.log_tabular('ActionError', action_error)
                self.traces_logger.log_tabular('GoalError', goal_error)

            if self.reward_curiosity:
                self.traces_logger.log_tabular('Curiosity', curiosity)
            self.traces_logger.dump_tabular(file_only=True)

        return lambda sess, ep_stopped, ep_done: self.handle_path_termination(sess, ep_stopped, ep_done, goal, observations, reward, stability)

    def handle_path_termination(self, sess, ep_stopped, ep_done, goal, observations, reward, stability):
        if ep_stopped:
            # if trajectory didn't reach terminal state, bootstrap value target
            if ep_done:
                last_val = stability + reward
            else:
                last_val = sess.run([self.value], feed_dict={self.x_ph: observations.reshape(1,-1), self.goals_ph: [goal]})

            self.ppo_buf.finish_path(last_val)

            self.prev_goal = None

        elif self.finish_action_path_on_new_goal:
            # Finish paths for actions based on goals and not episodes.  Stability rewards for later goals should not
            # add to the rewards in the current goal.  This leads all goals to attenuate to most stable state.

            if goal != self.prev_goal:
                self.log('PathLen', self.ppo_buf.path_len())
                last_val = reward + stability
                self.ppo_buf.finish_path(last_val)

        self.prev_goal = goal

    def update(self, sess):

        # Train inverse and forward
        self.inverse_buf.append(self.trajectory_buf)
        inverse_inputs = \
            {k:v for k,v in zip([self.x_ph, self.x_next_ph, self.goals_ph, self.actions_ph], self.inverse_buf.get())}

        for _ in range(self.train_inverse_iters):
            sess.run(self.train_inverse, feed_dict=inverse_inputs)
        inverse_action_loss_t, inverse_goal_loss_t = \
            sess.run([self.inverse_action_loss, self.inverse_goal_loss], feed_dict=inverse_inputs)

        if self.reward_curiosity:
            for _ in range(self.train_inverse_iters):
                sess.run(self.train_forward, feed_dict=inverse_inputs)
            forward_loss_t  = sess.run([self.forward_loss], feed_dict=inverse_inputs)

        # Prepare PPO training inputs
        policy_inputs = {k:v for k,v in zip([self.x_ph, self.x_next_ph, self.goals_ph, self.actions_ph], self.trajectory_buf.get())}

        actions_ppo_inputs, adv_mean = self.ppo_buf.get()
        policy_inputs.update({k:v for k,v in zip([self.adv_ph, self.ret_ph, self.logp_old_ph], actions_ppo_inputs)})

        # Train actions
        pi_loss_old, value_loss_old, entropy = sess.run([self.pi_loss, self.value_loss, self.approx_ent], feed_dict=policy_inputs)

        stop_iter = core.train_ppo(sess, self.train_pi_iters, self.train_pi, self.approx_kl, self.target_kl, policy_inputs)
        self.log('StopIter', stop_iter)

        for _ in range(self.train_value_iters):
            sess.run(self.train_value, feed_dict=policy_inputs)

        self.trajectory_buf.reset()

        self.log('LossPi', pi_loss_old)
        self.log('LossValue', value_loss_old)
        self.log('AdvantageMean', adv_mean)
        self.log('Entropy', entropy)
        self.log('LossActionInverse', inverse_action_loss_t)
        self.log('LossGoalInverse', inverse_goal_loss_t)
        if self.reward_curiosity:
            self.log('LossForward', forward_loss_t)

    def log_epoch(self):
        if self.log_level > 0:
            self.log_tabular('StopIter', average_only=True)
            if self.reward_stability:
                self.log_tabular('PathLen', average_only=True)
                self.log_tabular('ActionError', average_only=True)
                self.log_tabular('GoalError', average_only=True)
            if self.log_level > 1:
                self.log_tabular('LossPi', average_only=True)
                self.log_tabular('LossValue', average_only=True)
                self.log_tabular('AdvantageMean', average_only=True)
                self.log_tabular('VVals', average_only=True)
                self.log_tabular('Entropy', average_only=True)
                self.log_tabular('Reward', average_only=True)
                self.log_tabular('StabilityReward', average_only=True)
                self.log_tabular('LossActionInverse', average_only=True)
                self.log_tabular('LossGoalInverse', average_only=True)
                if self.reward_curiosity:
                    self.log_tabular('LossForward', average_only=True)
                    self.log_tabular('Curiosity', average_only=True)



def goaly(
        # Environment and policy
        env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(),
        steps_per_epoch=4000, epochs=50, max_ep_len=1000,
        # Goals
        goal_octaves=3, goal_error_base=1,
        goals_gamma=0.9, goals_clip_ratio=0.2, goals_pi_lr=3e-4, goals_vf_lr=1e-3,
        train_goals_pi_iters=80, train_goals_v_iters=80, goals_lam=0.97, goals_target_kl=0.01,
        # Actions
        goaly_kwargs=dict(),
        # Inverse model
        inverse_kwargs=dict(), train_inverse_iters=20, inverse_lr=1e-3,
        inverse_buffer_size=2,
        # Reward Calculations
        no_step_reward=False, no_path_len_reward=False,
        # etc.
        logger_kwargs=dict(), save_freq=10, seed=0, trace_freq=5):

    """

    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: A function which takes in placeholder symbols
            for state, ``x_ph`` concatenated with ``goals_ph``, and action, ``a_ph``, and returns the main
            outputs from the agent's Tensorflow computation graph:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``pi``       (batch, act_dim)  | Samples actions from policy given
                                           | states.
            ``logp``     (batch,)          | Gives log probability, according to
                                           | the policy, of taking actions ``a_ph``
                                           | in states ``x_ph``.
            ``logp_pi``  (batch,)          | Gives log probability, according to
                                           | the policy, of the action sampled by
                                           | ``pi``.
            ``v``        (batch,)          | Gives the value estimate for states
                                           | in ``x_ph``. (Critical: make sure
                                           | to flatten this!)
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the actor_critic
            function you provided to PPO.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs)
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs of interaction (equivalent to
            number of policy updates) to perform.

        gamma (float): Discount factor. (Always between 0 and 1.)

        clip_ratio (float): Hyperparameter for clipping in the policy objective.
            Roughly: how far can the new policy go from the old policy while
            still profiting (improving the objective function)? The new policy
            can still go farther than the clip_ratio says, but it doesn't help
            on the objective anymore. (Usually small, 0.1 to 0.3.)

        action_pi_lr (float): Learning rate for policy optimizer.

        action_vf_lr (float): Learning rate for value function optimizer.

        train_actions_pi_iters (int): Maximum number of gradient descent steps to take
            on policy loss per epoch. (Early stopping may cause optimizer
            to take fewer than this.)

        train_actions_v_iters (int): Number of gradient descent steps to take on
            value function per epoch.

        lam (float): Lambda for GAE-Lambda. (Always between 0 and 1,
            close to 1.)

        max_ep_len (int): Maximum length of trajectory / episode / rollout.

        target_kl (float): Roughly what KL divergence we think is appropriate
            between new and old policies after an update. This will get used
            for early stopping. (Usually small, 0.01 or 0.05.)

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

    """

    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    traces_logger = Logger(output_fname='traces.txt', **logger_kwargs)

    seed += 10000 * proc_id()
    tf.set_random_seed(seed)
    np.random.seed(seed)

    env = env_fn()
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    num_goals = 2**goal_octaves
    local_steps_per_epoch = int(steps_per_epoch / num_procs())

    # Inputs to computation graph
    x_ph, x_next_ph, actions_ph = core.placeholders_from_env(env)

    actions_policy = GoalyPolicy("Actions", logger, logger_kwargs, trace_freq, env.action_space, env.observation_space,
                                 x_ph, x_next_ph, actions_ph, num_goals, actor_critic, steps_per_epoch,
                                 **goaly_kwargs.get('Actions', dict()))

    goals_ph = tf.placeholder(dtype=tf.int32, shape=(None, ), name="goals_ph")

    goals2_ph = tf.placeholder(dtype=tf.int32, shape=(None, ), name="goals2_ph")

    goals_policy = GoalyPolicy("Goals", logger, logger_kwargs, trace_freq, Discrete(num_goals), env.observation_space,
                               x_ph, x_next_ph, goals_ph, num_goals, actor_critic, steps_per_epoch,
                                **goaly_kwargs.get('Goals', dict()))

    goals2_policy = GoalyPolicy("Goals2", logger, logger_kwargs, trace_freq, Discrete(num_goals), env.observation_space,
                               x_ph, x_next_ph, goals_ph, num_goals, actor_critic, steps_per_epoch,
                                **goaly_kwargs.get('Goals2', dict()))

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    # TODO: get outputs to save from policy classes
    # logger.setup_tf_saver(sess, inputs={'x': x_ph, 'goals_ph': goals_ph }, \
    #                       outputs={'pi': actions_pi, 'v': actions_v, 'goals_pi': goals_pi, 'goals_v': goals_v})

    def update():

        # Train policies
        goals2_policy.update(sess)
        goals_policy.update(sess)
        actions_policy.update(sess)

    start_time = time.time()
    observations, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    goal = 0

    ep_obs = [[]]
    episode = 0

    def handle_episode_termination(episode, goal, prev_goal, observations, reward, ep_done, ep_ret, ep_len):
        terminal = ep_done or (ep_len == max_ep_len)
        if terminal or (t==local_steps_per_epoch-1):
            episode += 1
            if not(terminal):
                print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)

            if terminal:
                # only save EpRet / EpLen if trajectory finished
                logger.store(EpRet=ep_ret, EpLen=ep_len)

            observations, reward, ep_done, ep_ret, ep_len = env.reset(), 0, False, 0, 0

        return episode, observations, reward, ep_done, ep_ret, ep_len

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            # Every step, get: action, value, and logprob
            prev_goal = goal

            goal2, goals2_store = goals2_policy.get(sess, observations, 0)
            goal, goals_store = goals_policy.get(sess, observations, goal2)
            actions, actions_store = actions_policy.get(sess, observations, goal)

            new_observations, reward, done, _ = env.step(actions)

            ep_ret += reward
            ep_len += 1

            handle_goals2_path_termination = goals2_store(epoch, episode, reward, new_observations)
            handle_goals_path_termination = goals_store(epoch, episode, reward, new_observations)
            handle_actions_path_termination = actions_store(epoch, episode, reward, new_observations)

            observations = new_observations
            ep_stopped = done or (ep_len == max_ep_len) or (t==local_steps_per_epoch-1)

            handle_goals2_path_termination(sess, ep_stopped, done)
            handle_goals_path_termination(sess, ep_stopped, done)
            handle_actions_path_termination(sess, ep_stopped, done)
            episode, observations, reward, done, ep_ret, ep_len = \
                handle_episode_termination(episode, goal, prev_goal, observations, reward, done, ep_ret, ep_len)

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet')
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('ActionsReward', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)

        goals2_policy.log_epoch()
        goals_policy.log_epoch()
        actions_policy.log_epoch()

        logger.dump_tabular()

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
    parser.add_argument('--exp_name', type=str, default='goaly')
    args = parser.parse_args()

    mpi_fork(args.cpu)  # run parallel code with mpi

    from spinup.utils.run_utils import setup_logger_kwargs
    logger_kwargs = setup_logger_kwargs(args.exp_name, args.seed)

    goaly(lambda : gym.make(args.env), actor_critic=core.mlp_actor_critic,
        ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), gamma=args.gamma,
        seed=args.seed, steps_per_epoch=args.steps, epochs=args.epochs,
        logger_kwargs=logger_kwargs)
