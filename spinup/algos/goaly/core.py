import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete, MultiBinary
from spinup.utils.mpi_tools import mpi_avg


EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None, name=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None, dim), name=name)

def placeholders(*args):
    return [placeholder(info[0], info[1]) for info in args]

def placeholders_goals(dim, env):
    if isinstance(env.observation_space, Box):
        return placeholder(dim)
    elif isinstance(env.observation_space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None, ))
    raise NotImplementedError

def placeholder_from_space(space, name):
    if isinstance(space, Box):
        return placeholder(space.shape, name)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,), name=name)
    raise NotImplementedError

def placeholders_from_env(env):
    observations_ph = placeholder_from_space(env.observation_space, "observations")
    next_observations_ph = placeholder_from_space(env.observation_space, "next_observations")
    actions_ph = placeholder_from_space(env.action_space, "actions")
    return observations_ph, next_observations_ph, actions_ph

def hidden(x, hidden_sizes=(32,), activation=tf.tanh):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return x

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    x = hidden(x, hidden_sizes=hidden_sizes, activation=activation)
    return tf.layers.dense(x, units=hidden_sizes[-1], activation=output_activation)

def get_vars(scope=''):
    return [x for x in tf.trainable_variables() if scope in x.name]

def count_vars(scope=''):
    v = get_vars(scope)
    return sum([np.prod(var.shape.as_list()) for var in v])

def gaussian_likelihood(x, mu, log_std):
    pre_sum = -0.5 * (((x-mu)/(tf.exp(log_std)+EPS))**2 + 2*log_std + np.log(2*np.pi))
    return tf.reduce_sum(pre_sum, axis=1)

def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.

    input:
        vector x,
        [x0,
         x1,
         x2]

    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]


"""
Policies
"""

def mlp_categorical_policy(x, goals, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = action_space.n
    if goals is None:
        features = x
    else:
        features = tf.concat([x, goals], 1)
    logits = mlp(features, list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi

def mlp_gaussian_policy(x, goals, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    if goals is None:
        features = x
    else:
        features = tf.concat([x, goals], 1)
    mu = mlp(features, list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi

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
def train_ppo(sess, train_iters, train_pi, approx_kl, target_kl, inputs):
    for i in range(train_iters):
        _, kl = sess.run([train_pi, approx_kl], feed_dict=inputs)
        kl = mpi_avg(kl)
        if kl > 1.5 * target_kl:
            # logger.log('Early stopping at step %d due to reaching max kl.'%i)
            break
    return i

"""
Actor-Critics
"""
def mlp_actor_critic(x, goals, a, hidden_sizes=(64,64), activation=tf.tanh,
                     output_activation=None, policy=None, action_space=None):

    # default policy builder depends on action space
    if policy is None and isinstance(action_space, Box):
        policy = mlp_gaussian_policy
    elif policy is None and isinstance(action_space, Discrete):
        policy = mlp_categorical_policy

    with tf.variable_scope('pi'):
        pi, logp, logp_pi = policy(x, goals, a, hidden_sizes, activation, output_activation, action_space)
    with tf.variable_scope('v'):
        if goals is None:
            features = x
        else:
            features = tf.concat([x, goals], 1)

        v = tf.squeeze(mlp(features, list(hidden_sizes)+[1], activation, None), axis=1)

    return pi, logp, logp_pi, v

"""
Inverse Dynamics
"""

def action_activation(x):
    return tf.round(x)

def inverse_model(action_space, x, x_next, a, goals, num_goals, hidden_sizes=(32,32), activation=tf.nn.relu, goals_output_activation=tf.sigmoid, inverse_buffer_size=3):
    inverse_input_size = tf.shape(x)[0]
    features_shape = x.shape.as_list()[1:]

    if isinstance(action_space, Discrete):
        # Convert 1/0 actions into one-hot actions.  This allows model to learn action values separately intead of
        # picking a value between two actions (like .5)
        a = tf.one_hot(tf.cast(a, tf.int32), 2)
        a_dim = np.prod(a.get_shape().as_list()[1:])
        a = tf.reshape(a, [-1, a_dim])

        actions_output_activation=tf.sigmoid
    else:
        actions_output_activation=None

    goals_one_hot = tf.one_hot(goals, num_goals)

    num_actions = a.get_shape().as_list()[-1]

    x = tf.concat([x_next, x], 1)
    hidden_x = hidden(x, list(hidden_sizes), activation)
    action_logits = mlp(hidden_x, [num_actions], activation, actions_output_activation)
    goals_logits = mlp(hidden_x, [num_goals], activation, goals_output_activation)

    goals_predicted = tf.argmax(goals_logits, axis=-1, output_type=tf.int32)

    return a, goals_one_hot, action_logits, goals_logits, goals_predicted

"""
Forward Dynamics
"""
def forward_model(x, a, x_next, is_action_space_discrete, hidden_sizes=(16,), activation=tf.nn.relu):
    features_shape = x.shape.as_list()[1:]

    if is_action_space_discrete:
        # Convert 1/0 actions into one-hot actions.  This allows model to learn action values separately intead of
        # picking a value between two actions (like .5)
        a = tf.one_hot(tf.cast(a, tf.int32), 2)
        a_dim = np.prod(a.get_shape().as_list()[1:])
        a = tf.reshape(a, [-1, a_dim])

    return mlp(tf.concat([x, a], 1), list(hidden_sizes)+features_shape, activation, None)

"""
Returns the high-low for action values, used to normalize action error in loss.
"""
def space_range(space):
    if isinstance(space, Discrete):
        range = 1
    else:
        range = space.high - space.low
    return range

"""
Goal Calculations
"""
