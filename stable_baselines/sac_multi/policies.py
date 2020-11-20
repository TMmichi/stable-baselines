from abc import abstractmethod

import tensorflow as tf
import numpy as np
from gym.spaces import Box

from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy
from stable_baselines.common.tf_layers import mlp

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
# CAP the standard deviation of the actor
LOG_STD_MAX = 2
LOG_STD_MIN = -20
debug = False


def gaussian_likelihood(input_, mu_, log_std):
    """
    Helper to computer log likelihood of a gaussian.
    Here we assume this is a Diagonal Gaussian.

    :param input_: (tf.Tensor)
    :param mu_: (tf.Tensor)
    :param log_std: (tf.Tensor)
    :return: (tf.Tensor)
    """
    pre_sum = -0.5 * (((input_ - mu_) / (tf.exp(log_std) + EPS)) ** 2 + 2 * log_std + np.log(2 * np.pi))
    return tf.reduce_sum(pre_sum, axis=1)


def gaussian_entropy(log_std):
    """
    Compute the entropy for a diagonal Gaussian distribution.

    :param log_std: (tf.Tensor) Log of the standard deviation
    :return: (tf.Tensor)
    """
    return tf.reduce_sum(log_std + 0.5 * np.log(2.0 * np.pi * np.e), axis=-1)


def clip_but_pass_gradient(input_, lower=-1., upper=1.):
    clip_up = tf.cast(input_ > upper, tf.float32)
    clip_low = tf.cast(input_ < lower, tf.float32)
    return input_ + tf.stop_gradient((upper - input_) * clip_up + (lower - input_) * clip_low)


def apply_squashing_func(mu_, pi_, logp_pi):
    """
    Squash the output of the Gaussian distribution
    and account for that in the log probability
    The squashed mean is also returned for using
    deterministic actions.

    :param mu_: (tf.Tensor) Mean of the gaussian
    :param pi_: (tf.Tensor) Output of the policy before squashing
    :param logp_pi: (tf.Tensor) Log probability before squashing
    :return: ([tf.Tensor])
    """
    # Squash the output
    deterministic_policy = tf.tanh(mu_)
    policy = tf.tanh(pi_)
    # OpenAI Variation:
    # To avoid evil machine precision error, strictly clip 1-pi**2 to [0,1] range.
    # logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - policy ** 2, lower=0, upper=1) + EPS), axis=1)
    # Squash correction (from original implementation)
    logp_pi -= tf.reduce_sum(tf.log(1 - policy ** 2 + EPS), axis=1)
    return deterministic_policy, policy, logp_pi


def fuse_networks_MCP(mu_array, log_std_array, weight, act_index, total_action_dimension):
    """
    Fuse distributions of policy into a MCP fashion

    :param mu_array: ([tf.Tensor]) List of means
    :param log_std_array: ([tf.Tensor]) List of log of the standard deviations
    :param weight: (tf.Tensor) Weight tensor of each primitives
    :param act_index: (list) List of action indices for each primitives
    :param total_action_dimension: (int) Dimension of a total action
    :return: ([tf.Tensor]) Samples of fused policy, fused mean, and fused standard deviations
    """
    with tf.variable_scope("fuse"):
        mu_MCP = std_sum = tf.tile(tf.reshape(weight[:,0],[-1,1]), tf.constant([1,total_action_dimension])) * 0
        if debug:
            mu_MCP = tf.Print(mu_MCP,[mu_MCP, tf.shape(mu_MCP)], "temp = ", summarize=-1)
            mu_MCP = tf.Print(mu_MCP,[weight, tf.shape(weight)], "weight = ", summarize=-1)
        for i in range(len(mu_array)):
            if debug:
                mu_array[i] = tf.Print(mu_array[i],[mu_array[i], tf.shape(mu_array[i])], "mu value {0} = ".format(i), summarize=-1)
                
            weight_tile_index = tf.tile(tf.reshape(weight[:,i],[-1,1]), tf.constant([1,mu_array[i][0].shape[0].value]))
            normed_weight_index = tf.math.divide_no_nan(weight_tile_index, tf.exp(log_std_array[i]))
            mu_weighted_i = mu_array[i] * normed_weight_index
            if debug:
                weight_tile_index = tf.Print(weight_tile_index, [weight_tile_index, tf.shape(weight_tile_index)], "weight_tile {0} = ".format(i), summarize=-1)
                mu_weighted_i = tf.Print(mu_weighted_i,[mu_weighted_i], "weighted mu value {0} = ".format(i), summarize=-1)
            append_idx = 0
            for j in range(total_action_dimension):
                print("Primitive Index ", i)
                if j in act_index[i]:
                    print("\tIn act index")
                    if j == 0:
                        mu_temp = tf.reshape(mu_weighted_i[:,append_idx], [-1,1], name="mu_temp")
                        std_temp = tf.reshape(normed_weight_index[:,append_idx], [-1,1], name="std_temp")
                    else:
                        mu_temp = tf.concat([mu_temp, tf.reshape(mu_weighted_i[:,append_idx], [-1,1])], 1, name="mu_temp")
                        std_temp = tf.concat([std_temp, tf.reshape(normed_weight_index[:,append_idx], [-1,1])], 1, name="std_temp")
                    append_idx += 1
                    if debug:
                        mu_temp = tf.Print(mu_temp, [mu_temp],"mu_temp(in) at {0}, {1} = ".format(i,j), summarize=-1)
                else:
                    print("\tNot in act index")
                    if j == 0:
                        mu_temp = tf.reshape(mu_weighted_i[:,0]*0, [-1,1], name="mu_temp")
                        std_temp = tf.reshape(normed_weight_index[:,0]*0, [-1,1], name="std_temp")
                    else:
                        mu_temp = tf.concat([mu_temp, tf.reshape(mu_weighted_i[:,0]*0, [-1,1])], 1, name="mu_temp")
                        std_temp = tf.concat([std_temp, tf.reshape(normed_weight_index[:,0]*0, [-1,1])], 1, name="std_temp")
                    if debug:
                        mu_temp = tf.Print(mu_temp, [mu_temp],"mu_temp(not in) at {0}, {1} = ".format(i,j), summarize=-1)
            mu_MCP += mu_temp
            std_sum += std_temp
            if debug:
                mu_MCP = tf.Print(mu_MCP, [mu_MCP],"mu_MCP at {0} = ".format(i), summarize=-1)
                std_sum = tf.Print(std_sum, [std_sum],"std_sum at {0} = ".format(i), summarize=-1)
        std_MCP = tf.math.reciprocal_no_nan(std_sum)
    mu_MCP = tf.math.multiply(mu_MCP, std_MCP, name="mu_MCP")
    if debug:
        mu_MCP = tf.Print(mu_MCP,[mu_MCP[0]],"mu_MCP = ")
    pi_MCP = tf.math.add(mu_MCP, tf.random_normal(tf.shape(mu_MCP)) * std_MCP, name="pi_MCP")
    
    log_std_MCP = tf.log(tf.clip_by_value(std_MCP,1e-10, 1e10), name="log_std_MCP")
    if debug:
        log_std_MCP = tf.Print(log_std_MCP,[log_std_MCP[0]],"log_std_MCP = ")
        log_std_MCP = tf.Print(log_std_MCP,[],"")
    
    return pi_MCP, mu_MCP, log_std_MCP


class SACPolicy(BasePolicy):
    """
    Policy object that implements a SAC-like actor critic

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param scale: (bool) whether or not to scale the input
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, scale=False, obs_phs=None):
        super(SACPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse, scale=scale, obs_phs=obs_phs)
        assert isinstance(ac_space, Box), "Error: the action space must be of type gym.spaces.Box"

        self.qf1 = None
        self.qf2 = None
        self.value_fn = None
        self.policy = None
        self.deterministic_policy = None
        self.act_mu = None
        self.std = None

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        """
        Creates an actor object

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name of the actor
        :return: (TensorFlow Tensor) the output tensor
        """
        raise NotImplementedError

    def make_critics(self, obs=None, action=None, reuse=False,
                     scope="values_fn", create_vf=True, create_qf=True):
        """
        Creates the two Q-Values approximator along with the Value function

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param action: (TensorFlow Tensor) The action placeholder
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name
        :param create_vf: (bool) Whether to create Value fn or not
        :param create_qf: (bool) Whether to create Q-Values fn or not
        :return: ([tf.Tensor]) Mean, action and log probability
        """
        raise NotImplementedError

    @abstractmethod
    def make_custom_actor(self, primitives, obs=None, reuse=False, scope="pi"):
        """
        Creates an custom actor object

        :param primitives: (dict) Obs/act information of primitives
        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name of the actor
        :return: (TensorFlow Tensor) the output tensor
        """
        raise NotImplementedError

    @abstractmethod
    def make_custom_critics(self, primitives, obs=None, action=None, separate_value=True, reuse=False,
                    scope="values_fn", create_vf=True, create_qf=True):
        """
        Creates the two Q-Values approximator along with the custom Value function

        :param primitives: (dict) Obs/act information of primitives
        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param action: (TensorFlow Tensor) The action placeholder
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name
        :param create_vf: (bool) Whether to create Value fn or not
        :param create_qf: (bool) Whether to create Q-Values fn or not
        :return: ([tf.Tensor]) Mean, action and log probability
        """
        raise NotImplementedError

    def step(self, obs, state=None, mask=None, deterministic=False):
        """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: ([float]) actions
        """
        raise NotImplementedError

    def proba_step(self, obs, state=None, mask=None):
        """
        Returns the action probability params (mean, std) for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float], [float])
        """
        raise NotImplementedError


class FeedForwardPolicy(SACPolicy):
    """
    Policy object that implements a DDPG-like actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param layer_norm: (bool) enable layer normalisation
    :param reg_weight: (float) Regularization loss weight for the policy parameters
    :param act_fun: (tf.func) the activation function to use in the neural network.
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers={}, obs_phs=None,
                 cnn_extractor=nature_cnn, feature_extraction="cnn", reg_weight=0.0,
                 layer_norm=False, act_fun=tf.nn.relu, **kwargs):
        super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                                reuse=reuse, scale=(feature_extraction == "cnn"), obs_phs=obs_phs)

        self._kwargs_check(feature_extraction, kwargs)
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.reuse = reuse
        self.policy_layers = layers.get('policy',[64,64])
        self.value_layers = layers.get('value',[64,64])
        self.weight = {}
        self.primitive_actions = {}
        self.primitive_log_std = {}
        self.primitive_value = {}
        self.primitive_qf1 = {}
        self.primitive_qf2 = {}
        self.reg_loss = None
        self.reg_weight = reg_weight
        self.entropy = 0

        self.activ_fn = act_fun

    def make_actor(self, obs=None, reuse=False, scope="pi"):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                pi_h = tf.layers.flatten(obs)

            pi_h = mlp(pi_h, self.policy_layers, self.activ_fn, layer_norm=self.layer_norm)

            self.act_mu = self.primitive_actions['mu_'] = mu_ = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)
            # Important difference with SAC and other algo such as PPO:
            # the std depends on the state, so we cannot use stable_baselines.common.distribution
            log_std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)
            self.primitive_log_std['std'] = log_std

        # Regularize policy output (not used for now)
        # reg_loss = self.reg_weight * 0.5 * tf.reduce_mean(log_std ** 2)
        # reg_loss += self.reg_weight * 0.5 * tf.reduce_mean(mu ** 2)
        # self.reg_loss = reg_loss

        # OpenAI Variation to cap the standard deviation
        # activation = tf.tanh # for log_std
        # log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        # Original Implementation
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std = std = tf.exp(log_std)
        # Reparameterization trick
        pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * std
        logp_pi = gaussian_likelihood(pi_, mu_, log_std)
        self.entropy = gaussian_entropy(log_std)
        # MISSING: reg params for log and mu
        # Apply squashing and account for it in the probability
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi

    def make_critics(self, obs=None, action=None, reuse=False, scope="values_fn",
                     create_vf=True, create_qf=True):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if self.feature_extraction == "cnn":
                critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                critics_h = tf.layers.flatten(obs)

            if create_vf:
                # Value function
                with tf.variable_scope('vf', reuse=reuse):
                    vf_h = mlp(critics_h, self.value_layers, self.activ_fn, layer_norm=self.layer_norm)
                    value_fn = tf.layers.dense(vf_h, 1, name="vf")
                self.value_fn = value_fn

            if create_qf:
                # Concatenate preprocessed state and action
                qf_h = tf.concat([critics_h, action], axis=-1)

                # Double Q values to reduce overestimation
                with tf.variable_scope('qf1', reuse=reuse):
                    qf1_h = mlp(qf_h, self.value_layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf1 = tf.layers.dense(qf1_h, 1, name="qf1")

                with tf.variable_scope('qf2', reuse=reuse):
                    qf2_h = mlp(qf_h, self.value_layers, self.activ_fn, layer_norm=self.layer_norm)
                    qf2 = tf.layers.dense(qf2_h, 1, name="qf2")

                self.qf1 = qf1
                self.qf2 = qf2

        return self.qf1, self.qf2, self.value_fn

    def make_custom_actor(self, obs=None, primitives=None, tails=None, total_action_dimension=0, reuse=False, scope="pi"):
        """
        Creates a custom actor object

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param primitives: (dict) Obs/act information of primitives
        :param total_action_dimension: (int) Dimension of a total action
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name of the actor
        :return: (TensorFlow Tensor) the output tensor
        """
        if obs is None:
            obs = self.processed_obs
        with tf.variable_scope(scope, reuse=reuse):
            pi_MCP, mu_MCP, log_std_MCP = self.construct_actor_graph(obs, primitives, tails, total_action_dimension, reuse)

        logp_pi = gaussian_likelihood(pi_MCP, mu_MCP, log_std_MCP)
        self.std = tf.exp(log_std_MCP)
        self.policy_train = pi_MCP
        self.deterministic_policy_train = self.act_mu = mu_MCP

        # policies with squashing func at test time
        # TODO: Need to check if these variables are used @ training time
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_MCP, pi_MCP, logp_pi)
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi

    def make_custom_critics(self, obs=None, action=None, primitives=None, tails=None, scope="values_fn", reuse=False, 
                    create_vf=True, create_qf=True):
        """
        Creates the two Q-Values approximator along with the custom Value function

        :param primitives: (dict) Obs/act information of primitives
        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param action: (TensorFlow Tensor) The action placeholder
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name
        :param create_vf: (bool) Whether to create Value fn or not
        :param create_qf: (bool) Whether to create Q-Values fn or not
        :return: ([tf.Tensor]) Mean, action and log probability
        """
        self.qf1 = 0
        self.qf2 = 0
        self.value_fn = 0

        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            if create_vf:
                with tf.variable_scope('vf', reuse=reuse):
                    self.construct_value_graph(obs, action, primitives, tails, reuse=reuse, create_vf=True)
            
            if create_qf:
                # Double Q values to reduce overestimation
                with tf.variable_scope('qf1', reuse=reuse):
                    self.construct_value_graph(obs, action, primitives, tails, reuse=reuse, create_qf=True, qf1=True)
                with tf.variable_scope('qf2', reuse=reuse):
                    self.construct_value_graph(obs, action, primitives, tails, reuse=reuse, create_qf=True, qf2=True)

        return self.qf1, self.qf2, self.value_fn

    def construct_actor_graph(self, obs=None, primitives=None, tails=None, total_action_dimension=0, reuse=False):
        print("Received tails in actor graph: ",tails)
        if obs is None:
            obs = self.processed_obs

        mu_array = []
        log_std_array = []
        act_index = []
        weight = None

        for name in tails:
            item = primitives[name]
            layer_name = item['layer_name'] if item['main_tail'] else name
            if item['tails'] == None:
                if 'weight' in name.split('/'):
                    with tf.variable_scope(layer_name, reuse=reuse):
                        if self.feature_extraction == "cnn":
                            pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
                            raise NotImplementedError("Image input not supported for now")
                        else:
                            pi_h = tf.layers.flatten(obs)
                        
                        #------------- Input observation seiving layer -------------#
                        seive_layer = np.zeros([obs.shape[1].value, len(item['obs'][1])], dtype=np.float32)
                        for i in range(len(item['obs'][1])):
                            seive_layer[item['obs'][1][i]][i] = 1
                        pi_h = tf.matmul(pi_h, seive_layer)
                        #------------- Observation seiving layer End -------------#

                        pi_h = mlp(pi_h, item['layer']['policy'], self.activ_fn, layer_norm=self.layer_norm)
                        weight = tf.layers.dense(pi_h, len(item['act'][1]), activation='softmax')
                        self.weight[name] = weight
                else:
                    with tf.variable_scope(layer_name, reuse=reuse):
                        if self.feature_extraction == "cnn":
                            pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
                            raise NotImplementedError("Image input not supported for now")
                        else:
                            pi_h = tf.layers.flatten(obs)
                        
                        #------------- Input observation seiving layer -------------#
                        seive_layer = np.zeros([obs.shape[1].value, len(item['obs'][1])], dtype=np.float32)
                        for i in range(len(item['obs'][1])):
                            seive_layer[item['obs'][1][i]][i] = 1
                        pi_h = tf.matmul(pi_h, seive_layer)
                        #------------- Observation seiving layer End -------------#

                        pi_h = mlp(pi_h, item['layer']['policy'], self.activ_fn, layer_norm=self.layer_norm)

                        mu_ = tf.layers.dense(pi_h, len(item['act'][1]), activation=None)
                        mu_array.append(mu_)
                        self.primitive_actions[name] = mu_

                        log_std = tf.layers.dense(pi_h, len(item['act'][1]), activation=None)
                        act_index.append(item['act'][1])

                    log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
                    log_std_array.append(log_std)
                    self.primitive_log_std[name] = log_std
                    
                    self.entropy += gaussian_entropy(log_std)
            else:
                with tf.variable_scope(layer_name, reuse=reuse):
                    _, mu_, log_std_ = self.construct_actor_graph(obs, primitives, item['tails'], total_action_dimension, reuse)
                mu_array.append(mu_)
                log_std_array.append(log_std_)
                act_index.append(item['act'][1])
        
        assert not isinstance(weight, type(None)), \
            '\n\t\033[91m[ERROR]: No weight within tail:{0}\033[0m'.format(tails)
        pi_MCP, mu_MCP, log_std_MCP = fuse_networks_MCP(mu_array, log_std_array, weight, act_index, total_action_dimension)
        
        return pi_MCP, mu_MCP, log_std_MCP
   
    def construct_value_graph(self, obs=None, action=None, primitives=None, tails=None, reuse=False, 
                                create_vf=False, create_qf=False, qf1=False, qf2=False):
        print("Received tails in value graph: ",tails)
        if obs is None:
            obs = self.processed_obs

        for name in tails:
            item = primitives[name]
            if item['load_value']:
                layer_name = item['layer_name'] if item['main_tail'] else name
                if item['tails'] == None:
                    with tf.variable_scope(layer_name, reuse=reuse):
                        if self.feature_extraction == "cnn":
                            critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
                            raise NotImplementedError("Image input not supported for now")
                        else:
                            critics_h = tf.layers.flatten(obs)

                        #------------- Input observation seiving layer -------------#
                        seive_layer = np.zeros([obs.shape[1].value, len(item['obs'][1])], dtype=np.float32)
                        for i in range(len(item['obs'][1])):
                            seive_layer[item['obs'][1][i]][i] = 1
                        critics_h = tf.matmul(critics_h, seive_layer)
                        #------------- Observation seiving layer End -------------#

                        if create_vf:
                            # Value function
                            vf_h = mlp(critics_h, item['layer']['value'], self.activ_fn, layer_norm=self.layer_norm)
                            value_fn = tf.layers.dense(vf_h, 1, name="vf")
                            self.value_fn += value_fn
                            self.primitive_value[layer_name] = value_fn

                        if create_qf:
                            #------------- Input action seiving layer -------------#
                            seive_layer = np.zeros([action.shape[1], len(item['act'][1])], dtype=np.float32)
                            for i in range(len(item['act'][1])):
                                seive_layer[item['act'][1][i]][i] = 1
                            qf_h = tf.matmul(action, seive_layer)
                            #------------- Action seiving layer End -------------#
                            
                            # Concatenate preprocessed state and action
                            qf_h = tf.concat([critics_h, qf_h], axis=-1)

                            qf_h = mlp(qf_h, item['layer']['value'], self.activ_fn, layer_norm=self.layer_norm)
                            if qf1:
                                qf = tf.layers.dense(qf_h, 1, name="qf1")
                                self.qf1 += qf
                                self.primitive_qf1[layer_name] = qf
                            if qf2:
                                qf = tf.layers.dense(qf_h, 1, name="qf2")
                                self.qf2 += qf
                                self.primitive_qf2[layer_name] = qf
                else:
                    with tf.variable_scope(layer_name, reuse=reuse):
                        self.construct_value_graph(obs, action, primitives, item['tails'], reuse, create_vf, create_qf, qf1, qf2)
            if 'weight' in name.split('/'):
                composite_name = name.split('/')[0]
                layer_name = 'train/'+composite_name if item['main_tail'] else composite_name
                with tf.variable_scope(layer_name, reuse=reuse):
                    if self.feature_extraction == "cnn":
                        critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
                        raise NotImplementedError("Image input not supported for now")
                    else:
                        critics_h = tf.layers.flatten(obs)

                    #------------- Input observation seiving layer -------------#
                    seive_layer = np.zeros([obs.shape[1].value, len(item['obs'][1])], dtype=np.float32)
                    for i in range(len(item['obs'][1])):
                        seive_layer[item['obs'][1][i]][i] = 1
                    critics_h = tf.matmul(critics_h, seive_layer)
                    #------------- Observation seiving layer End -------------#

                    if create_vf:
                        # Value function
                        vf_h = mlp(critics_h, item['layer']['value'], self.activ_fn, layer_norm=self.layer_norm)
                        value_fn = tf.layers.dense(vf_h, 1, name="vf")
                        self.value_fn += value_fn
                        self.primitive_value[layer_name] = value_fn

                    if create_qf:
                        #------------- Input action seiving layer -------------#
                        seive_layer = np.zeros([action.shape[1], len(item['composite_action_index'])], dtype=np.float32)
                        for i in range(len(item['composite_action_index'])):
                            seive_layer[item['composite_action_index']][i] = 1
                        qf_h = tf.matmul(action, seive_layer)
                        #------------- Action seiving layer End -------------#
                        
                        # Concatenate preprocessed state and action
                        qf_h = tf.concat([critics_h, qf_h], axis=-1)

                        qf_h = mlp(qf_h, item['layer']['value'], self.activ_fn, layer_norm=self.layer_norm)
                        if qf1:
                            qf = tf.layers.dense(qf_h, 1, name="qf1")
                            self.qf1 += qf
                            self.primitive_qf1[layer_name] = qf
                        if qf2:
                            qf = tf.layers.dense(qf_h, 1, name="qf2")
                            self.qf2 += qf
                            self.primitive_qf2[layer_name] = qf

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run(self.deterministic_policy, {self.obs_ph: obs})
        return self.sess.run(self.policy, {self.obs_ph: obs})
    
    def get_weight(self, obs):
        return self.sess.run(self.weight, {self.obs_ph: obs})

    def get_primitive_action(self, obs):
        return self.sess.run(self.primitive_actions, {self.obs_ph: obs})

    def get_primitive_log_std(self, obs):
        return self.sess.run(self.primitive_log_std, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run([self.act_mu, self.std], {self.obs_ph: obs})


class CnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers={}, obs_phs=None, **_kwargs):
        super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, layers, obs_phs=obs_phs, 
                                        feature_extraction="cnn", **_kwargs)


class LnCnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers={}, obs_phs=None, **_kwargs):
        super(LnCnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, layers, obs_phs=obs_phs, 
                                          feature_extraction="cnn", layer_norm=True, **_kwargs)


class MlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers={}, obs_phs=None, **_kwargs):
        super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, layers, obs_phs=obs_phs, 
                                        feature_extraction="mlp", **_kwargs)


class LnMlpPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a MLP (2 layers of 64), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, reuse=False, layers={}, obs_phs=None, **_kwargs):
        super(LnMlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse, layers, obs_phs=obs_phs, 
                                          feature_extraction="mlp", layer_norm=True, **_kwargs)


register_policy("CnnPolicy", CnnPolicy)
register_policy("LnCnnPolicy", LnCnnPolicy)
register_policy("MlpPolicy", MlpPolicy)
register_policy("LnMlpPolicy", LnMlpPolicy)
