import numpy as np
import tensorflow as tf
import scipy.signal
from gym.spaces import Box, Discrete

EPS = 1e-8

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None, dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholders_goals(dim, env):
    if isinstance(env.observation_space, Box):
        return placeholder(dim)
    elif isinstance(env.observation_space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None, ))
    raise NotImplementedError

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_sizes=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_sizes[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
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
    logits = mlp(tf.concat([x, goals], 1), list(hidden_sizes)+[act_dim], activation, None)
    logp_all = tf.nn.log_softmax(logits)
    pi = tf.squeeze(tf.multinomial(logits,1), axis=1)
    logp = tf.reduce_sum(tf.one_hot(a, depth=act_dim) * logp_all, axis=1)
    logp_pi = tf.reduce_sum(tf.one_hot(pi, depth=act_dim) * logp_all, axis=1)
    return pi, logp, logp_pi

def mlp_gaussian_policy(x, goals, a, hidden_sizes, activation, output_activation, action_space):
    act_dim = a.shape.as_list()[-1]
    mu = mlp(tf.concat([x, goals], 1), list(hidden_sizes)+[act_dim], activation, output_activation)
    log_std = tf.get_variable(name='log_std', initializer=-0.5*np.ones(act_dim, dtype=np.float32))
    std = tf.exp(log_std)
    pi = mu + tf.random_normal(tf.shape(mu)) * std
    logp = gaussian_likelihood(a, mu, log_std)
    logp_pi = gaussian_likelihood(pi, mu, log_std)
    return pi, logp, logp_pi

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
        v = tf.squeeze(mlp(x, list(hidden_sizes)+[1], activation, None), axis=1)
    return pi, logp, logp_pi, v

"""
Inverse Dynamics
"""

def action_activation(x):
    return tf.round(x)

def inverse_model(env, x, a, goals, hidden_sizes=(32,32), activation=tf.nn.relu):

    inverse_input_size = tf.shape(x)[0]
    features_shape = x.shape.as_list()[1:]
    x_prev = tf.slice(x, [0, 0], [inverse_input_size - 1] + features_shape)
    x = tf.slice(x, [1, 0], [inverse_input_size - 1] + features_shape)

    if isinstance(env.action_space, Discrete):
        # Trim the last action from input set
        a_inverse = tf.slice(a, [0], [inverse_input_size - 1])

        # Convert 1/0 actions into one-hot actions.  This allows model to learn action values separately intead of
        # picking a value between two actions (like .5)
        a_inverse = tf.one_hot(tf.cast(a_inverse, tf.int32), 2)
        a_inverse_dim = np.prod(a_inverse.get_shape().as_list()[1:])
        a_inverse = tf.reshape(a_inverse, [-1, a_inverse_dim])

        output_activation=tf.sigmoid
    else:
        # Trim the last action from input set
        a_inverse = tf.slice(a, [0, 0], [inverse_input_size - 1] + list(env.action_space.shape))
        output_activation=None

    num_goals = goals.get_shape().as_list()[-1]
    goals_inverse = tf.slice(goals, [0, 0], [inverse_input_size - 1, num_goals])

    num_actions = a_inverse.get_shape().as_list()[-1]

    logits = mlp(tf.concat([x_prev, x], 1), list(hidden_sizes)+[num_actions + num_goals], activation, output_activation)

    inverse_input_size = tf.shape(x)[0]
    action_logits = tf.slice(logits, [0, 0], [inverse_input_size, num_actions])
    goals_logits = tf.slice(logits, [0, num_actions], [inverse_input_size, num_goals])
    return a_inverse, action_logits, goals_logits
