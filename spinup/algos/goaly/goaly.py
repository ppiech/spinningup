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

    def __init__(self, obs_dim, act_dim, size):
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

    def get(self, reset=True):
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

        if reset:
            self.ptr = 0

        return to_return

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
                    insert_at = np.random.randint(0, self.max_size)
                else:
                    insert_at = self.ptr
                    self.ptr += 1

                self.obs_buf[insert_at] = new_obs[mpi_proc_num][i]
                self.new_obs_buf[insert_at] = new_new_obs[mpi_proc_num][i]
                self.goals_buf[insert_at] = new_goals[mpi_proc_num][i]
                self.act_buf[insert_at] = new_act[mpi_proc_num][i]

def ppo_objectives(adv_ph, val, ret_ph, logp, logp_old_ph, clip_ratio):
    ratio = tf.exp(logp - logp_old_ph)          # pi(a|s) / pi_old(a|s)
    min_adv = tf.where(adv_ph>0, (1+clip_ratio)*adv_ph, (1-clip_ratio)*adv_ph)
    pi_loss = -tf.reduce_mean(tf.minimum(ratio * adv_ph, min_adv))
    v_loss = tf.reduce_mean((ret_ph - val)**2)

    approx_kl = tf.reduce_mean(logp_old_ph - logp)      # a sample estimate for KL-divergence, easy to compute
    approx_ent = tf.reduce_mean(-logp)                  # a sample estimate for entropy, also easy to compute
    clipped = tf.logical_or(ratio > (1+clip_ratio), ratio < (1-clip_ratio))
    clipfrac = tf.reduce_mean(tf.cast(clipped, tf.float32))

    return pi_loss, v_loss, approx_kl, approx_ent, clipfrac

# Training
def train_ppo(train_iters, train_pi, approx_kl, target_kl):
    for i in range(train_iters):
        _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
        kl = mpi_avg(kl)
        if kl > 1.5 * target_kl:
            # logger.log('Early stopping at step %d due to reaching max kl.'%i)
            break
    return i

class ActionsPolicy:

    def __init__(self, logger, env, x_ph, x_next_ph, actions_ph, num_goals, actor_critic, steps_per_epoch, actions_gamma,
                 actions_lam, actions_clip_ratio, ac_kwargs, inverse_kwargs):
        self.logger = logger
        self.x_ph, self.x_next_ph, self.actions_ph = x_ph, x_next_ph, actions_ph

        self.actions_adv_ph = core.placeholder(None, "actions_adv_ph")
        self.actions_ret_ph = core.placeholder(None, "actions_ret_ph")
        self.actions_logp_old_ph = core.placeholder(None, "actions_logp_old_ph")
        self.goals_ph = tf.placeholder(dtype=tf.int32, shape=(None, ), name="goals_ph")

        with tf.variable_scope('actions'):
            self.goals_pi_actions_input = tf.one_hot(self.goals_ph, num_goals)
            self.actions_pi, self.actions_logp, self.actions_logp_pi, self.actions_v = actor_critic(
                self.x_ph, self.goals_pi_actions_input, self.actions_ph, action_space=env.action_space, **ac_kwargs)

        obs_dim = env.observation_space.shape
        act_dim = env.action_space.shape

        local_steps_per_epoch = int(steps_per_epoch / num_procs())
        self.actions_ppo_buf = PPOBuffer(local_steps_per_epoch, actions_gamma, actions_lam)
        self.inverse_buf = ObservationsActionsAndGoalsBuffer(obs_dim, act_dim, steps_per_epoch*inverse_kwargs['invese_buffer_size'])

        # Count variables
        var_counts = tuple(core.count_vars(scope) for scope in ['actions_pi', 'actions_v'])
        logger.log('\nNumber of parameters: \t actions_pi: %d, \t actions_v: %d\n'%var_counts)

        self.actions_pi_loss, self.actions_v_loss, self.actions_approx_kl, self.actions_approx_ent, self.actions_clipfrac = ppo_objectives(
            self.actions_adv_ph, self.actions_v, self.actions_ret_ph, self.actions_logp, self.actions_logp_old_ph, actions_clip_ratio)

        # Inverse Dynamics Model
        self.a_inverse, self.goals_one_hot, self.a_predicted, self.goals_predicted_logits, self.goals_predicted = \
            core.inverse_model(env, self.x_ph, self.x_next_ph, self.actions_ph, self.goals_ph, num_goals, **inverse_kwargs)
        a_range = core.space_range(env.action_space)

        a_as_float = tf.cast(a_inverse, tf.float32)

        inverse_action_diff = tf.abs((a_as_float - a_predicted) / a_range, name='inverse_action_diff')
        inverse_action_loss = tf.reduce_mean((a_as_float - a_predicted)**2, name='inverse_action_loss') # For training inverse model

        # if isinstance(env.action_space, Discrete):
            # inverse_action_diff = tf.reshape(tf.reduce_mean(inverse_action_diff, axis=1), [-1, 1], 'inverse_action_diff')
        inverse_goal_loss = tf.reduce_mean((tf.cast(goals_one_hot - goals_predicted_logits, tf.float32)**2), name='inverse_goal_loss')
        self.inverse_loss = inverse_goal_loss + inverse_action_loss

        # Errors used for calculating return after each step.
        # Action error needs to be normalized wrt action amplitude, otherwise the error will drive the model behavior
        # towards small amplitude actions.  This doesn't apply to discrete action spaces, where a_predicted is a set of
        # logits compared againt a_inverse that is a one-hot vector.
        if isinstance(env.action_space, Discrete):
            inverse_action_error_denominator = 1.0
        else:
            inverse_action_error_denominator = tf.math.maximum(((tf.abs(a_as_float + a_predicted)) * 10 / a_range), 1e-4)

        self.inverse_action_error = tf.reduce_mean(inverse_action_diff / inverse_action_error_denominator)

        # when calculating goal error for stability reward, compare numerical goal value
        self.inverse_goal_error = tf.reduce_mean(tf.abs(tf.cast(goals_predicted - goals_ph, tf.float32)) * 2.0 / num_goals)

        # by default use invese action error in stability reward
        self.stability_reward = 1 + 2*self.inverse_action_error * self.inverse_goal_error - self.inverse_action_error - self.inverse_goal_error

        # cap stability reward
        self.stability_reward =  tf.math.maximum(tf.math.minimum(stability_reward, 1.0), -1.0)

        self.train_actions_pi = MpiAdamOptimizer(learning_rate=action_pi_lr).minimize(actions_pi_loss)
        self.train_actions_v = MpiAdamOptimizer(learning_rate=action_vf_lr).minimize(actions_v_loss)

        self.train_inverse = MpiAdamOptimizer(learning_rate=inverse_lr).minimize(inverse_loss)

    def get(self, observations, goal):
        actions, actions_v_t, actions_logp_t = \
            sess.run([self.actions_pi, self.actions_v, self.actions_logp_pi],
                     feed_dict={x_ph: observations.reshape(1,-1), self.goals_ph: goal })
        return actions, actions_v_t, actions_logp_t

    def store(self, observations, new_observations, goal, goals_step_reward, actions,
                    actions_reward_v, reward, stability, actions_v_t, forward_prediction_error):
        if not forward_error_for_curiosity_reward:
            forward_prediction_error = 0

        if actions_step_reward:
            actions_ppo_buf.store(reward, actions_reward_v, actions_v_t, actions_logp_t)
        else:
            actions_ppo_buf.store(reward + actions_reward_v, 0, actions_v_t, actions_logp_t)

        self.logger.store(ActionsVVals=actions_v_t)
        self.logger.store(Goal=goal)
        # logger.storeOne("Goal{}Reward".format(goal), reward)
        self.logger.store(GoalPathLen=actions_ppo_buf.path_len())

    def update(self, sess, logger, trajectory_buf, train_inverse_iters):
        # Train inverse and forward
        self.inverse_buf.append(trajectory_buf)
        self.inverse_inputs = {k:v for k,v in zip([self.x_ph, self.x_next_ph, self.goals_ph, self.actions_ph], inverse_buf.get(reset=False))}
        inverse_action_loss_old, inverse_goal_loss_old  = sess.run([inverse_action_loss, inverse_goal_loss], feed_dict=inverse_inputs)

        for _ in range(train_inverse_iters):
            sess.run(train_inverse, feed_dict=inverse_inputs)

        logger.store(LossActionInverse=inverse_action_loss_new)
        logger.store(DeltaLossActionInverse=(inverse_action_loss_new - inverse_action_loss_old))
        logger.store(LossGoalInverse=inverse_goal_loss_new)
        logger.store(DeltaLossGoalInverse=(inverse_goal_loss_new - inverse_goal_loss_old))        # logger.store(LossForward=forward_loss_new)


def goaly(
        # Environment and policy
        env_fn, actor_critic=core.mlp_actor_critic, ac_kwargs=dict(),
        steps_per_epoch=4000, epochs=50, max_ep_len=1000,
        # Goals
        goal_octaves=3, goal_error_base=1,
        goals_gamma=0.9, goals_clip_ratio=0.2, goals_pi_lr=3e-4, goals_vf_lr=1e-3,
        train_goals_pi_iters=80, train_goals_v_iters=80, goals_lam=0.97, goals_target_kl=0.01,
        # Actions
        actions_gamma=0.99, actions_lam=0.97, actions_clip_ratio=0.2, action_pi_lr=3e-4, action_vf_lr=1e-3,
        train_actions_pi_iters=80, train_actions_v_iters=80, actions_target_kl=0.01,
        # Inverse model
        inverse_kwargs=dict(), train_inverse_iters=20, inverse_lr=1e-3,
        invese_buffer_size=2,
        # Reward Calculations
        finish_action_path_on_new_goal=True, no_step_reward=False,
        forward_error_for_curiosity_reward=False,  actions_step_reward=False, no_path_len_reward=False,
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
    # actions_adv_ph = core.placeholder(None, "actions_adv_ph")
    # actions_ret_ph = core.placeholder(None, "actions_ret_ph")
    # actions_logp_old_ph = core.placeholder(None, "actions_logp_old_ph")

    actions_policy = ActionsPolicy(logger, env, x_ph, x_next_ph, actions_ph, num_goals, actor_critic, steps_per_epoch,
                                   actions_gamma, actions_lam, actions_clip_ratio, ac_kwargs, inverse_kwargs)

    goals_adv_ph = core.placeholder(None, "goals_adv_ph")
    goals_ret_ph = core.placeholder(None, "goals_ret_ph")
    goals_logp_old_ph = core.placeholder(None, "goals_logp_old_ph")

    goals_ph = tf.placeholder(dtype=tf.int32, shape=(None, ), name="goals_ph")

    # Main outputs from computation graph
    with tf.variable_scope('goals'):
        goals_pi, goals_logp, goals_logp_pi, goals_v = actor_critic(
            x_ph, None, goals_ph, action_space=Discrete(num_goals), **ac_kwargs)

    # with tf.variable_scope('actions'):
    #     goals_pi_actions_input = tf.one_hot(goals_ph, num_goals)
    #     actions_pi, actions_logp, actions_logp_pi, actions_v = actor_critic(
    #         x_ph, goals_pi_actions_input, actions_ph, action_space=env.action_space, **ac_kwargs)

    # Experience buffer
    goals_ppo_buf = PPOBuffer(local_steps_per_epoch, goals_gamma, goals_lam)
    # actions_ppo_buf = PPOBuffer(local_steps_per_epoch, actions_gamma, actions_lam)
    trajectory_buf = ObservationsActionsAndGoalsBuffer(obs_dim, act_dim, local_steps_per_epoch)
    # inverse_buf = ObservationsActionsAndGoalsBuffer(obs_dim, act_dim, steps_per_epoch*invese_buffer_size)

    # # Count variables
    # var_counts = tuple(core.count_vars(scope) for scope in ['actions_pi', 'actions_v'])
    # logger.log('\nNumber of parameters: \t actions_pi: %d, \t actions_v: %d\n'%var_counts)

    # # Inverse Dynamics Model
    # a_inverse, goals_one_hot, a_predicted, goals_predicted_logits, goals_predicted = \
    #     core.inverse_model(env, x_ph, x_next_ph, actions_ph, goals_ph, num_goals, **inverse_kwargs)
    # a_range = core.space_range(env.action_space)
    #
    # a_as_float = tf.cast(a_inverse, tf.float32)
    #
    # inverse_action_diff = tf.abs((a_as_float - a_predicted) / a_range, name='inverse_action_diff')
    #
    # # if isinstance(env.action_space, Discrete):
    #     # inverse_action_diff = tf.reshape(tf.reduce_mean(inverse_action_diff, axis=1), [-1, 1], 'inverse_action_diff')
    # inverse_goal_loss = tf.reduce_mean((tf.cast(goals_one_hot - goals_predicted_logits, tf.float32)**2), name='inverse_goal_loss')
    # inverse_loss = inverse_goal_loss + inverse_action_loss
    #
    # # Errors used for calculating return after each step.
    # # Action error needs to be normalized wrt action amplitude, otherwise the error will drive the model behavior
    # # towards small amplitude actions.  This doesn't apply to discrete action spaces, where a_predicted is a set of
    # # logits compared againt a_inverse that is a one-hot vector.
    # if isinstance(env.action_space, Discrete):
    #     inverse_action_error_denominator = 1.0
    # else:
    #     inverse_action_error_denominator = tf.math.maximum(((tf.abs(a_as_float + a_predicted)) * 10 / a_range), 1e-4)
    #
    # inverse_action_error = tf.reduce_mean(inverse_action_diff / inverse_action_error_denominator)
    #
    # # when calculating goal error for stability reward, compare numerical goal value
    # inverse_goal_error = tf.reduce_mean(tf.abs(tf.cast(goals_predicted - goals_ph, tf.float32)) * 2.0 / num_goals)

    # Forward model
    is_action_space_discrete = isinstance(env.action_space, Discrete)
    x_pred = core.forward_model(x_ph, actions_ph, x_next_ph, is_action_space_discrete)

    forward_diff = tf.abs((tf.cast(x_ph, tf.float32) - x_pred), name='forward_diff')
    forward_error = tf.reduce_mean(forward_diff, name="forward_error")
    forward_loss = tf.reduce_mean((forward_error)**2, name="forward_loss")

    goals_pi_loss, goals_v_loss, goals_approx_kl, goals_approx_ent, goals_clipfrac = ppo_objectives(
        goals_adv_ph, goals_v, goals_ret_ph, goals_logp, goals_logp_old_ph, goals_clip_ratio)

    # actions_pi_loss, actions_v_loss, actions_approx_kl, actions_approx_ent, actions_clipfrac = ppo_objectives(
    #     actions_adv_ph, actions_v, actions_ret_ph, actions_logp, actions_logp_old_ph, actions_clip_ratio)

    # goaly reward

    # # by default use invese action error in stability reward
    # stability_reward = 1 + 2*inverse_action_error * inverse_goal_error - inverse_action_error - inverse_goal_error
    #
    # # cap stability reward
    # stability_reward =  tf.math.maximum(tf.math.minimum(stability_reward, 1.0), -1.0)

    # Optimizers
    train_goals_pi = MpiAdamOptimizer(learning_rate=goals_pi_lr).minimize(goals_pi_loss)
    train_goals_v = MpiAdamOptimizer(learning_rate=goals_vf_lr).minimize(goals_v_loss)
    # train_actions_pi = MpiAdamOptimizer(learning_rate=action_pi_lr).minimize(actions_pi_loss)
    # train_actions_v = MpiAdamOptimizer(learning_rate=action_vf_lr).minimize(actions_v_loss)
    #
    # train_inverse = MpiAdamOptimizer(learning_rate=inverse_lr).minimize(inverse_loss)
    train_forward = MpiAdamOptimizer(learning_rate=inverse_lr).minimize(forward_loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Sync params across processes
    sess.run(sync_all_params())

    # Setup model saving
    # TODO: get outputs to save from policy classes
    # logger.setup_tf_saver(sess, inputs={'x': x_ph, 'goals_ph': goals_ph }, \
    #                       outputs={'pi': actions_pi, 'v': actions_v, 'goals_pi': goals_pi, 'goals_v': goals_v})

    def update():

        # # Train inverse and forward
        # inverse_buf.append(trajectory_buf)
        # inverse_inputs = {k:v for k,v in zip([x_ph, x_next_ph, goals_ph, actions_ph], inverse_buf.get(reset=False))}
        # inverse_action_loss_old, inverse_goal_loss_old  = sess.run([inverse_action_loss, inverse_goal_loss], feed_dict=inverse_inputs)
        #
        # for _ in range(train_inverse_iters):
        #     sess.run(train_inverse, feed_dict=inverse_inputs)

        # Train forward
        inverse_inputs = {k:v for k,v in zip([x_ph, x_next_ph, goals_ph, actions_ph], inverse_buf.get(reset=False))}
        forward_loss_old  = sess.run([forward_loss], feed_dict=inverse_inputs)
        for _ in range(train_inverse_iters):
            sess.run(train_forward, feed_dict=inverse_inputs)

        # Prepare PPO training inputs
        inputs = {k:v for k,v in zip([x_ph, x_next_ph, goals_ph, actions_ph], trajectory_buf.get())}

        actions_ppo_inputs, actions_adv_mean = actions_ppo_buf.get()
        inputs.update({k:v for k,v in zip([actions_adv_ph, actions_ret_ph, actions_logp_old_ph], actions_ppo_inputs)})

        goals_ppo_inputs, goals_adv_mean = goals_ppo_buf.get()
        inputs.update({k:v for k,v in zip([goals_adv_ph, goals_ret_ph, goals_logp_old_ph], goals_ppo_inputs)})

        # # Training
        # def train_ppo(train_iters, train_pi, approx_kl, target_kl):
        #     for i in range(train_iters):
        #         _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
        #         kl = mpi_avg(kl)
        #         if kl > 1.5 * target_kl:
        #             # logger.log('Early stopping at step %d due to reaching max kl.'%i)
        #             break
        #     return i

        # Train goals
        goals_pi_l_old, goals_v_l_old, goals_ent = sess.run([goals_pi_loss, goals_v_loss, goals_approx_ent], feed_dict=inputs)
        stop_iter = train_ppo(train_goals_pi_iters, train_goals_pi, goals_approx_kl, goals_target_kl)
        logger.store(GoalsStopIter=stop_iter)

        for _ in range(train_goals_v_iters):
            sess.run(train_goals_v, feed_dict=inputs)

        # Train actions
        actions_pi_l_old, actions_v_l_old, actions_ent = sess.run([actions_pi_loss, actions_v_loss, actions_approx_ent], feed_dict=inputs)
        stop_iter = train_ppo(train_actions_pi_iters, train_actions_pi, actions_approx_kl, actions_target_kl)
        logger.store(ActionsStopIter=stop_iter)

        for _ in range(train_actions_v_iters):
            sess.run(train_actions_v, feed_dict=inputs)

        # Log changes from update
        actions_pi_l_new, actions_v_l_new,  actions_kl, actions_cf, \
            goals_pi_l_new, goals_v_l_new, goals_kl, goals_cf, \
            inverse_action_loss_new, inverse_goal_loss_new, forward_loss_new = \
            sess.run([
                actions_pi_loss, actions_v_loss, actions_approx_kl, actions_clipfrac, \
                goals_pi_loss, goals_v_loss, goals_approx_kl, goals_clipfrac, \
                inverse_action_loss, inverse_goal_loss, forward_loss],
                feed_dict=inputs)

        logger.store(LossActionsPi=actions_pi_l_old)
        logger.store(LossActionsV=actions_v_l_old)
        logger.store(DeltaLossActionsPi=(actions_pi_l_new - actions_pi_l_old))
        logger.store(DeltaLossActionsV=(actions_v_l_new - actions_v_l_old))
        logger.store(ActionsAdvantageMean=actions_adv_mean)
        logger.store(ActionsKL=actions_kl)
        logger.store(ActionsEntropy=actions_ent)
        logger.store(ActionsClipFrac=actions_cf)
        logger.store(LossGoalsPi=goals_pi_l_old)
        logger.store(LossGoalsV=goals_v_l_old)
        logger.store(DeltaLossGoalsPi=(goals_pi_l_new - goals_pi_l_old))
        logger.store(DeltaLossGoalsV=(goals_v_l_new - goals_v_l_old))
        logger.store(GoalsAdvantageMean=goals_adv_mean)
        logger.store(GoalsKL=goals_kl)
        logger.store(GoalsEntropy=goals_ent)
        logger.store(GoalsClipFrac=goals_cf)
        # logger.store(LossActionInverse=inverse_action_loss_new)
        # logger.store(DeltaLossActionInverse=(inverse_action_loss_new - inverse_action_loss_old))
        # logger.store(LossGoalInverse=inverse_goal_loss_new)
        # logger.store(DeltaLossGoalInverse=(inverse_goal_loss_new - inverse_goal_loss_old))        # logger.store(LossForward=forward_loss_new)
        # logger.store(DeltaLossForward=(forward_loss_new - forward_loss_old))

    start_time = time.time()
    observations, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
    goal, stability, action_error, goal_error, forward_prediction_error, goal_predicted_v = 0, 0, 0, 0, 0, 0

    ep_obs = [[]]
    episode = 0

    def store_training_step(observations, new_observations, goal, goals_step_reward, actions,
                            actions_reward_v, reward, goals_v_t, goals_logp_t, stability, actions_v_t,
                            forward_prediction_error):
        trajectory_buf.store(observations, new_observations, goal, actions)

        if not forward_error_for_curiosity_reward:
            forward_prediction_error = 0

        if no_step_reward:
            goals_ppo_buf.store(reward + forward_prediction_error, 0, goals_v_t, goals_logp_t)
        else:
            goals_ppo_buf.store(reward + forward_prediction_error, goals_step_reward, goals_v_t, goals_logp_t)

        if actions_step_reward:
            actions_ppo_buf.store(reward, actions_reward_v, actions_v_t, actions_logp_t)
        else:
            actions_ppo_buf.store(reward + actions_reward_v, 0, actions_v_t, actions_logp_t)

        logger.store(ActionsVVals=actions_v_t)
        logger.store(GoalsVVals=goals_v_t)
        logger.store(Goal=goal)
        # logger.storeOne("Goal{}Reward".format(goal), reward)
        logger.store(GoalPathLen=actions_ppo_buf.path_len())

    def log_trace_step(epoch, episode, observations, actions, goal, reward, goals_v_t, \
                       goals_step_reward_v, actions_reward_v, action_error, goal_error, \
                       forward_prediction_error, goals_predicted_v):
        if episode % trace_freq == 0:
            traces_logger.log_tabular('Epoch', epoch)
            traces_logger.log_tabular('Episode', episode)
            for i in range(0, len(observations)):
                traces_logger.log_tabular('Observations{}'.format(i), observations[i])
            if isinstance(env.action_space, Discrete):
                traces_logger.log_tabular('Actions0'.format(i), actions)
            else:
                for i in range(0, len(actions)):
                    traces_logger.log_tabular('Actions{}'.format(i), actions[i])
            traces_logger.log_tabular('Reward', reward)
            traces_logger.log_tabular('Goal', goal)
            traces_logger.log_tabular('GoalPredicted', goals_predicted_v)
            traces_logger.log_tabular('GoalsVVal', goals_v_t)
            traces_logger.log_tabular('GoalsStepReward', goals_step_reward_v)
            traces_logger.log_tabular('ActionsReward', actions_reward_v)
            traces_logger.log_tabular('ActionError', action_error)
            traces_logger.log_tabular('GoalError', goal_error)
            traces_logger.log_tabular('ForwardPredictionError', forward_prediction_error)

            traces_logger.dump_tabular(file_only=True)

    def calculate_stability(observations, new_observations, actions, goal):
        stability, action_error, goal_error, forward_prediction_error, goals_predicted_v, x_pred_v = \
            sess.run([stability_reward, inverse_action_error, inverse_goal_error, forward_error, goals_predicted, x_pred],
                      feed_dict={x_ph: np.array([observations]),
                                 x_next_ph: np.array([new_observations]),
                                 actions_ph: np.array([actions]),
                                 goals_ph: np.array([goal])})
        logger.store(StabilityReward=stability, StabilityActionError=action_error, StabilityGoalError=goal_error, ForwardPredictionError=forward_prediction_error)

        return stability, action_error, goal_error, forward_prediction_error, int(goals_predicted_v[0])

    def handle_episode_termination(episode, goal, prev_goal, observations, reward, done, ep_ret, ep_len, stability):
        terminal = done or (ep_len == max_ep_len)
        if terminal or (t==local_steps_per_epoch-1):
            episode += 1
            if not(terminal):
                print('Warning: trajectory cut off by epoch at %d steps.'%ep_len)
            # if trajectory didn't reach terminal state, bootstrap value target
            if done:
                last_actions_val = actions_reward(reward, stability)
                last_goals_val = goals_step_reward(reward, stability) + reward
            else:
                last_actions_val, last_goals_val = \
                    sess.run([actions_v, goals_v],
                             feed_dict={x_ph: observations.reshape(1,-1), goals_ph: [goal] })

            actions_ppo_buf.finish_path(last_actions_val)
            goals_ppo_buf.finish_path(last_goals_val)

            if terminal:
                # only save EpRet / EpLen if trajectory finished
                logger.store(EpRet=ep_ret, EpLen=ep_len)

            observations, reward, done, ep_ret, ep_len = env.reset(), 0, False, 0, 0
        elif finish_action_path_on_new_goal:
            # Finish paths for actions based on goals and not episodes.  Stability rewards for later goals should not
            # add to the rewards in the current goal.  This leads all goals to attenuate to most stable state.

            if goal != prev_goal:
                logger.store(GoalPathLen=actions_ppo_buf.path_len())
                last_actions_val = actions_reward(reward, stability)
                # debug: get last path value from values model
                # last_actions_val = sess.run([actions_v], feed_dict={x_ph: observations.reshape(1,-1), goals_ph: [goal]})
                actions_ppo_buf.finish_path(last_actions_val)

        return episode, observations, reward, done, ep_ret, ep_len

    def actions_reward(reward, stability):
        r = stability
        logger.store(ActionsReward=r)
        return r

    def goals_step_reward(reward, stability):
        if no_path_len_reward:
            r = stability
        else:
            r = stability + actions_ppo_buf.path_len()

        logger.store(GoalsStepReward=r)
        return r

    # Main loop: collect experience in env and update/log each epoch
    for epoch in range(epochs):
        for t in range(local_steps_per_epoch):
            # Every step, get: action, value, and logprob
            prev_goal = goal

            goal, goals_v_t, goals_logp_t = \
                sess.run([goals_pi, goals_v, goals_logp_pi],
                         feed_dict={x_ph: observations.reshape(1,-1) })

            actions, actions_v_t, actions_logp_t = actions_policy.get(observations, goal)

            goal = goal[0]
            actions = actions[0]
            goals_v_t = goals_v_t[0]

            goals_step_reward_v = goals_step_reward(reward, stability)
            actions_reward_v = actions_reward(reward, stability)

            new_observations, new_reward, done, _ = env.step(actions)

            store_training_step(observations, new_observations, goal, goals_step_reward_v, actions,
                                actions_reward_v, reward, goals_v_t, goals_logp_t, stability, actions_v_t, forward_prediction_error)

            stability, action_error, goal_error, forward_prediction_error, goal_predicted_v = \
                calculate_stability(observations, new_observations, actions, goal)

            log_trace_step(epoch, episode, observations, actions, goal, new_reward,
                           goals_v_t, goals_step_reward_v, actions_reward_v, action_error, goal_error,
                           forward_prediction_error, goal_predicted_v)

            observations = new_observations
            reward = new_reward
            ep_ret += reward
            ep_len += 1

            episode, observations, reward, done, ep_ret, ep_len = \
                handle_episode_termination(episode, goal, prev_goal, observations, reward, done, ep_ret, ep_len, stability)

        # Save model
        if (epoch % save_freq == 0) or (epoch == epochs-1):
            logger.save_state({'env': env}, None)

        # Perform PPO update!
        update()

        # Log info about epoch
        logger.log_tabular('Epoch', epoch)
        logger.log_tabular('EpRet')
        logger.log_tabular('EpLen', average_only=True)
        logger.log_tabular('GoalPathLen', average_only=True)
        logger.log_tabular('StabilityReward', average_only=True)
        logger.log_tabular('StabilityActionError', average_only=True)
        logger.log_tabular('StabilityGoalError', average_only=True)
        logger.log_tabular('ForwardPredictionError', average_only=True)
        logger.log_tabular('GoalsStepReward', average_only=True)
        logger.log_tabular('ActionsReward', average_only=True)
        logger.log_tabular('TotalEnvInteracts', (epoch+1)*steps_per_epoch)
        logger.log_tabular('LossGoalsPi', average_only=True)
        logger.log_tabular('DeltaLossGoalsPi', average_only=True)
        logger.log_tabular('LossGoalsV', average_only=True)
        logger.log_tabular('DeltaLossGoalsV', average_only=True)
        logger.log_tabular('GoalsVVals', average_only=True)
        logger.log_tabular('GoalsEntropy', average_only=True)
        logger.log_tabular('GoalsAdvantageMean', average_only=True)
        logger.log_tabular('GoalsKL', average_only=True)
        logger.log_tabular('GoalsClipFrac', average_only=True)
        logger.log_tabular('GoalsStopIter', average_only=True)
        logger.log_tabular('LossActionsPi', average_only=True)
        logger.log_tabular('DeltaLossActionsPi', average_only=True)
        logger.log_tabular('LossActionsV', average_only=True)
        logger.log_tabular('DeltaLossActionsV', average_only=True)
        logger.log_tabular('ActionsVVals', average_only=True)
        logger.log_tabular('ActionsEntropy', average_only=True)
        logger.log_tabular('ActionsAdvantageMean', average_only=True)
        logger.log_tabular('ActionsKL', average_only=True)
        logger.log_tabular('ActionsClipFrac', average_only=True)
        logger.log_tabular('ActionsStopIter', average_only=True)
        logger.log_tabular('LossActionInverse', average_only=True)
        logger.log_tabular('DeltaLossActionInverse', average_only=True)
        logger.log_tabular('LossGoalInverse', average_only=True)
        logger.log_tabular('DeltaLossGoalInverse', average_only=True)
        logger.log_tabular('LossForward', average_only=True)
        logger.log_tabular('DeltaLossForward', average_only=True)
        # for goal_num in range(num_goals):
        #     logger.log_tabular("Goal{}Reward".format(goal_num), average_only=True)
        # logger.log_tabular('Time', time.time()-start_time)
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
