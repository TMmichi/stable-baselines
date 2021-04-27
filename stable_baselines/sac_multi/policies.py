from abc import abstractmethod

from numpy.lib.function_base import select

import tensorflow as tf
import numpy as np
from gym.spaces import Box

from stable_baselines.common.policies import BasePolicy, nature_cnn, register_policy
from stable_baselines.common.tf_layers import mlp

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
weight_EPS = 1e-10
# CAP the standard deviation of the actor
LOG_STD_MAX = 5
LOG_STD_MIN = -5
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
    # pre_sum = tf.Print(pre_sum, [pre_sum,],'\nlog likelihood: ', summarize=-1)
    # pre_sum = tf.Print(pre_sum, [tf.reduce_sum(pre_sum, axis=1),],'summed log likelihood: ', summarize=-1)
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
    logp_pi -= tf.reduce_sum(tf.log(clip_but_pass_gradient(1 - policy ** 2, lower=0, upper=1) + EPS), axis=1)
    # Squash correction (from original implementation)
    # logp_pi -= tf.reduce_sum(tf.log(1 - policy ** 2 + EPS), axis=1)
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
        mu_temp = std_sum = tf.tile(tf.reshape(weight[:,0],[-1,1]), tf.constant([1,total_action_dimension])) * 0
        total_weight_sum = tf.tile(tf.reshape(weight[:,0],[-1,1]), tf.constant([1,total_action_dimension])) * 0
        for i in range(len(mu_array)):
            weight_tile = tf.tile(tf.reshape(weight[:,i],[-1,1]), tf.constant([1,mu_array[i][0].shape[0].value]))
            normed_weight_index = tf.math.divide_no_nan(weight_tile, tf.exp(log_std_array[i]))
            mu_weighted_i = mu_array[i] * normed_weight_index
            shaper = np.zeros([len(act_index[i]), total_action_dimension], dtype=np.float32)
            for j, index in enumerate(act_index[i]):
                shaper[j][index] = 1
            mu_temp += tf.matmul(mu_weighted_i, shaper)
            std_sum += tf.matmul(normed_weight_index, shaper)
            # TEST:
            total_weight_sum += tf.matmul(weight_tile, shaper)
        std_MCP = tf.math.reciprocal_no_nan(std_sum)
        mu_MCP = tf.math.multiply(mu_temp, std_MCP, name="mu_MCP")
        # TEST:
        mu_MCP *= total_weight_sum
        std_MCP *= total_weight_sum
        # mu_MCP = tf.Print(mu_MCP, [total_weight_sum,], 'total weight sum: ', summarize=-1)
        log_std_MCP = tf.log(std_MCP, name="log_std_MCP")
        
        pi_MCP = tf.math.add(mu_MCP, tf.random_normal(tf.shape(mu_MCP)) * tf.exp(log_std_MCP), name="pi_MCP")
    
    return pi_MCP, mu_MCP, log_std_MCP

def fuse_networks_categorical(mu_array, log_std_array, weight, act_index, total_action_dimension):
    for i in range(len(mu_array)):
        shaper = np.zeros([len(act_index[i]), total_action_dimension], dtype=np.float32)
        for j, index in enumerate(act_index[i]):
            shaper[j][index] = 1
        mu_array[i] = tf.matmul(mu_array[i], shaper)
        log_std_array[i] = tf.matmul(log_std_array[i], shaper)

    logit = tf.log(weight/(1-weight))

    uniform = tf.random_uniform(tf.shape(logit), dtype=logit.dtype)
    select_idx = tf.argmax(logit - tf.log(-tf.log(uniform)), axis=-1) * 2
    
    def mu0(): return mu_array[0]
    def mu1(): return mu_array[1]
    def logstd0(): return log_std_array[0]
    def logstd1(): return log_std_array[1]
     
    mu_categorical = tf.cond(tf.reduce_mean(select_idx - 1) < 0, mu0, mu1)
    log_std_categorical = tf.cond(tf.reduce_mean(select_idx - 1) < 0, logstd0, logstd1)
    pi_categorical = tf.math.add(mu_categorical, tf.random_normal(tf.shape(mu_categorical)) * tf.exp(log_std_categorical), name="pi_categorical")
    return pi_categorical, mu_categorical, log_std_categorical, select_idx


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
    def make_HPC_actor(self, primitives, obs=None, reuse=False, scope="pi"):
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
    def make_HPC_critics(self, primitives, obs=None, action=None, separate_value=True, reuse=False,
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
        self.dist = None
        self.layer_norm = layer_norm
        self.feature_extraction = feature_extraction
        self.cnn_kwargs = kwargs
        self.cnn_extractor = cnn_extractor
        self.reuse = reuse
        if layers.get('policy', None) is None and kwargs.get('net_arch',[{}])[0].get('pi',None) is None:
            self.policy_layers = [64,64]
        else:
            self.policy_layers = layers.get('policy', None) if layers.get('policy', None) is not None else kwargs['net_arch'][0]['pi']
        if layers.get('value', None) is None and kwargs.get('net_arch',[{}])[0].get('vf',None) is None:
            self.value_layers = [64,64]
        else:
            self.value_layers = layers.get('value', None) if layers.get('value', None) is not None else kwargs['net_arch'][0]['vf']
        self.obs_relativity = kwargs.get('obs_relativity', {})
        self.obs_index = kwargs.get('obs_index',None)
        self.weight = {}
        self.weight_0 = {}
        self.weight_1 = {}
        self.weight_ph = {}
        self.subgoal = {}
        self.primitive_actions = {}
        self.primitive_log_std = {}
        self.primitive_value = {}
        self.primitive_qf1 = {}
        self.primitive_qf2 = {}
        self.reg_loss = None
        self.reg_weight = reg_weight
        self.activ_fn = act_fun
        self.selected_idx = None

    def make_actor_benchmark(self, obs=None, reuse=False, scope="agent/main/actor", non_log=False):
        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            obs_index = list(range(self.processed_obs.shape[1].value)) if self.obs_index is None else self.obs_index

            #------------- Input observation sieving layer -------------#
            index_pair = {}
            sieve_layer = np.zeros([self.processed_obs.shape[1].value, len(obs_index)], dtype=np.float32)
            for i in range(len(obs_index)):
                index_pair[obs_index[i]] = i
                sieve_layer[obs_index[i]][i] = 1
            print(sieve_layer.shape)
            sieved_obs = tf.matmul(tf.layers.flatten(self.processed_obs), sieve_layer)
            #------------- Observation sieving layer End -------------#

            new_obs = sieved_obs
            pi_h = tf.layers.flatten(new_obs)
            pi_h = tf.layers.dense(inputs=pi_h,
                                    units=1024,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=tf.nn.relu,
                                    name='0/dense')
            pi_h = tf.layers.dense(inputs=pi_h, 
                                    units=512,
                                    kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                    activation=None,
                                    name='1/dense')
            pi_h = tf.nn.relu(pi_h)

            self.act_mu = mu_ = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None, name='dist_gauss_diag/mean')
            # Important difference with SAC and other algo such as PPO:
            # the std depends on the state, so we cannot use stable_baselines.common.distribution
            bias_init = np.ones(self.ac_space.shape[0]).astype(np.float32)
            log_std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None, name='dist_gauss_diag/logstd')
            # log_std = tf.get_variable(dtype=tf.float32, name="dist_gauss_diag/logstd/bias", initializer=bias_init, trainable=False)
            # log_std = tf.broadcast_to(log_std, tf.shape(mu_))
            self.primitive_log_std['std'] = log_std

        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std = std = tf.exp(log_std)
        # Reparameterization trick
        pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * std
        logp_pi = gaussian_likelihood(pi_, mu_, log_std)
        self.entropy = gaussian_entropy(log_std)
        
        # Apply squashing and account for it in the probability
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_, pi_, logp_pi)
        
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi

    def make_actor(self, obs=None, reuse=False, scope="pi", non_log=False):
        if obs is None:
            obs = self.processed_obs

        non_log = False
        with tf.variable_scope(scope, reuse=reuse):
            obs_index = list(range(self.processed_obs.shape[1].value)) if self.obs_index is None else self.obs_index
            obs_relativity = {} if self.obs_relativity is None else self.obs_relativity

            #------------- Input observation sieving layer -------------#
            index_pair = {}
            sieve_layer = np.zeros([self.processed_obs.shape[1].value, len(obs_index)], dtype=np.float32)
            for i in range(len(obs_index)):
                index_pair[obs_index[i]] = i
                sieve_layer[obs_index[i]][i] = 1
            sieved_obs = tf.matmul(tf.layers.flatten(self.processed_obs), sieve_layer)
            #------------- Observation sieving layer End -------------#

            if 'subtract' in obs_relativity.keys():
                ref = obs_relativity['subtract']['ref']
                tar = obs_relativity['subtract']['tar']
                assert len(ref) == len(tar), "Error: length of reference and target indicies unidentical"
                ref_sieve = np.zeros([sieved_obs.shape[1].value, len(ref)], dtype=np.float32)
                tar_sieve = np.zeros([sieved_obs.shape[1].value, len(tar)], dtype=np.float32)
                remainder_list = list(range(sieved_obs.shape[1].value))
                for i in range(len(ref)):
                    remainder_list.remove(index_pair[ref[i]])
                    remainder_list.remove(index_pair[tar[i]])
                    ref_sieve[index_pair[ref[i]]][i] = 1
                    tar_sieve[index_pair[tar[i]]][i] = 1
                ref_obs = tf.matmul(sieved_obs, ref_sieve)
                tar_obs = tf.matmul(sieved_obs, tar_sieve)
                subs_obs = ref_obs - tar_obs
                if not len(remainder_list) == 0:
                    remainder = sieved_obs.shape[1].value - (len(ref) + len(tar))
                    left_sieve = np.zeros([sieved_obs.shape[1].value, remainder], dtype=np.float32)
                    for j in range(len(remainder_list)):
                        left_sieve[remainder_list[j]][j] = 1
                    rem_obs = tf.matmul(sieved_obs, left_sieve)
                    new_obs = tf.concat([subs_obs, rem_obs], axis=-1)
                else:
                    new_obs = subs_obs
                if 'leave' in obs_relativity.keys():
                    leave = obs_relativity['leave']
                    leave_sieve = np.zeros([sieved_obs.shape[1].value, len(leave)], dtype=np.float32)
                    for i in range(len(leave)):
                        leave_sieve[index_pair[leave[i]]][i] = 1
                    leave_obs = tf.matmul(sieved_obs, leave_sieve)
                    new_obs = tf.concat([new_obs, leave_obs], axis=-1)
            else:
                new_obs = sieved_obs
                
            # new_obs = tf.Print(new_obs, [new_obs,], 'new observation: ', summarize=-1)
            if self.feature_extraction == "cnn":
                pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                # pi_h = tf.layers.flatten(obs)
                # new_obs = tf.Print(new_obs, [new_obs,], 'subs obs', summarize=-1)
                pi_h = tf.layers.flatten(new_obs)
            # pi_h = tf.Print(pi_h, [pi_h,], 'obs: ', summarize=-1)
            pi_h = mlp(pi_h, self.policy_layers, self.activ_fn, layer_norm=self.layer_norm)

            self.act_mu = self.primitive_actions['mu_'] = mu_ = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)
            # Important difference with SAC and other algo such as PPO:
            # the std depends on the state, so we cannot use stable_baselines.common.distribution
            if not non_log:
                log_std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation=None)
            else:
                std = tf.layers.dense(pi_h, self.ac_space.shape[0], activation='relu') + EPS
                log_std = tf.log(std)
            self.primitive_log_std['std'] = log_std

        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        self.std = std = tf.exp(log_std)
        # Reparameterization trick
        pi_ = mu_ + tf.random_normal(tf.shape(mu_)) * std
        logp_pi = gaussian_likelihood(pi_, mu_, log_std)
        self.entropy = gaussian_entropy(log_std)
        
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
            obs_index = list(range(self.processed_obs.shape[1].value)) if self.obs_index is None else self.obs_index
            obs_relativity = {} if self.obs_relativity is None else self.obs_relativity

            #------------- Input observation sieving layer -------------#
            index_pair = {}
            sieve_layer = np.zeros([self.processed_obs.shape[1].value, len(obs_index)], dtype=np.float32)
            for i in range(len(obs_index)):
                index_pair[obs_index[i]] = i
                sieve_layer[obs_index[i]][i] = 1
            sieved_obs = tf.matmul(tf.layers.flatten(self.processed_obs), sieve_layer)
            #------------- Observation sieving layer End -------------#

            if 'subtract' in obs_relativity.keys():
                ref = obs_relativity['subtract']['ref']
                tar = obs_relativity['subtract']['tar']
                assert len(ref) == len(tar), "Error: length of reference and target indicies unidentical"
                ref_sieve = np.zeros([sieved_obs.shape[1].value, len(ref)], dtype=np.float32)
                tar_sieve = np.zeros([sieved_obs.shape[1].value, len(tar)], dtype=np.float32)
                remainder_list = list(range(sieved_obs.shape[1].value))
                for i in range(len(ref)):
                    remainder_list.remove(index_pair[ref[i]])
                    remainder_list.remove(index_pair[tar[i]])
                    ref_sieve[index_pair[ref[i]]][i] = 1
                    tar_sieve[index_pair[tar[i]]][i] = 1
                ref_obs = tf.matmul(sieved_obs, ref_sieve)
                tar_obs = tf.matmul(sieved_obs, tar_sieve)
                # ref_obs = tf.Print(ref_obs, [ref_obs,],"ref_obs: ", summarize=-1)
                # tar_obs = tf.Print(tar_obs, [tar_obs,],"tar_obs: ", summarize=-1)
                subs_obs = ref_obs - tar_obs
                # subs_obs = tf.Print(subs_obs,[subs_obs,],"subtracted_obs: ", summarize=-1)
                if not len(remainder_list) == 0:
                    remainder = sieved_obs.shape[1].value - (len(ref) + len(tar))
                    left_sieve = np.zeros([sieved_obs.shape[1].value, remainder], dtype=np.float32)
                    for j in range(len(remainder_list)):
                        left_sieve[remainder_list[j]][j] = 1
                    rem_obs = tf.matmul(sieved_obs, left_sieve)
                    new_obs = tf.concat([subs_obs, rem_obs], axis=-1)
                else:
                    new_obs = subs_obs
                if 'leave' in obs_relativity.keys():
                    leave = obs_relativity['leave']
                    leave_sieve = np.zeros([sieved_obs.shape[1].value, len(leave)], dtype=np.float32)
                    for i in range(len(leave)):
                        leave_sieve[index_pair[leave[i]]][i] = 1
                    leave_obs = tf.matmul(sieved_obs, leave_sieve)
                    new_obs = tf.concat([new_obs, leave_obs], axis=-1)
            else:
                new_obs = sieved_obs
                
            if self.feature_extraction == "cnn":
                critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
            else:
                # critics_h = tf.layers.flatten(obs)
                critics_h = tf.layers.flatten(new_obs)

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

                # qf1 = tf.Print(qf1,[qf1,],'qf1: ',summarize=-1)
                self.qf1 = qf1
                self.qf2 = qf2

        return self.qf1, self.qf2, self.value_fn

    def make_HPC_actor(self, obs=None, primitives=None, tails=None, total_action_dimension=0, reuse=False, loaded=False, scope="pi"):
        """
        Creates a HPC actor object

        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param primitives: (dict) Obs/act information of primitives
        :param total_action_dimension: (int) Dimension of a total action
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name of the actor
        :return: (TensorFlow Tensor) the output tensor
        """
        non_log = True
        self.weight_0_ph = tf.placeholder(tf.float32, shape=[], name='weight_0')
        self.weight_1_ph = tf.placeholder(tf.float32, shape=[], name='weight_1')

        if obs is None:
            obs = self.processed_obs
        with tf.variable_scope(scope, reuse=reuse):
            if loaded:
                pi_MCP, mu_MCP, log_std_MCP = \
                    self.construct_actor_graph(obs, primitives, tails, total_action_dimension, reuse, non_log)
            else:
                pi_MCP, mu_MCP, log_std_MCP, [pi_0, mu_0, log_std_0], [pi_1, mu_1, log_std_1], [pi_ph, mu_ph, log_std_ph] = \
                    self.construct_actor_graph(obs, primitives, tails, total_action_dimension, reuse, non_log)
                logp_pi_0 = gaussian_likelihood(pi_0, mu_0, log_std_0)
                logp_pi_1 = gaussian_likelihood(pi_1, mu_1, log_std_1)
                logp_pi_ph = gaussian_likelihood(pi_ph, mu_ph, log_std_ph)
                _, policy_0, logp_pi_0 = apply_squashing_func(mu_0, pi_0, logp_pi_0)
                _, policy_1, logp_pi_1 = apply_squashing_func(mu_1, pi_1, logp_pi_1)
                _, policy_ph, logp_pi_ph = apply_squashing_func(mu_ph, pi_ph, logp_pi_ph)
                self.policy_0 = policy_0
                self.policy_1 = policy_1
                self.policy_ph = policy_ph

        logp_pi = gaussian_likelihood(pi_MCP, mu_MCP, log_std_MCP)
        
        self.entropy = gaussian_entropy(log_std_MCP)
        self.std = tf.exp(log_std_MCP)

        # policies with squashing func at test time
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_MCP, pi_MCP, logp_pi)
        
        # weight_val = [self.weight[name] for name in self.weight.keys()]
        # policy = tf.Print(policy,[mu_MCP, self.std, pi_MCP, logp_pi, weight_val], "mu, std, pi, logpi, weight: ", summarize=-1)
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi
 
    def make_HPC_critics(self, obs=None, action=None, primitives=None, tails=None, scope="values_fn", reuse=False, 
                    create_vf=True, create_qf=True, weight=False, SACD=True):
        """
        Creates the two Q-Values approximator along with the HPC Value function

        :param primitives: (dict) Obs/act information of primitives
        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param action: (TensorFlow Tensor) The action placeholder
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name
        :param create_vf: (bool) Whether to create Value fn or not
        :param create_qf: (bool) Whether to create Q-Values fn or not
        :param weight: (bool) Whether to create Q-Values with respect to weights
        :param SACD: (bool) Q function for Discrete-SAC
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
                    self.construct_value_graph(obs, action, primitives, tails, reuse=reuse, create_qf=True, qf1=True, weight=weight, SACD=SACD)
                with tf.variable_scope('qf2', reuse=reuse):
                    self.construct_value_graph(obs, action, primitives, tails, reuse=reuse, create_qf=True, qf2=True, weight=weight, SACD=SACD)

        return self.qf1, self.qf2, self.value_fn

    def construct_actor_graph(self, obs=None, primitives=None, tails=None, total_action_dimension=0, reuse=False, non_log=False):
        print("Received tails in actor graph: ",tails)
        if obs is None:
            obs = self.processed_obs

        mu_array = []
        log_std_array = []
        act_index = []
        weight = None
        weight_biased = False

        # Meta-controller Initialization
        for name in tails:
            if 'weight' in name.split('/'):
                print('Graph of '+name+' initializing')
                prim_dict = primitives[name]
                main_tail = True if prim_dict['main_tail'] else False
                layer_name = prim_dict['layer_name'] if main_tail else name
                with tf.variable_scope(layer_name, reuse=reuse):
                    if self.feature_extraction == "cnn":
                        pi_h = self.cnn_extractor(obs, **self.cnn_kwargs)
                        raise NotImplementedError("Image input not supported for now")
                    else:
                        pi_h = tf.layers.flatten(obs)
                    
                    #------------- Input observation sieving layer -------------#
                    sieve_layer = np.zeros([obs.shape[1].value, len(prim_dict['obs'][1])], dtype=np.float32)
                    for i in range(len(prim_dict['obs'][1])):
                        sieve_layer[prim_dict['obs'][1][i]][i] = 1
                    pi_h = tf.matmul(pi_h, sieve_layer)
                    #------------- Observation sieving layer End -------------#
                    pi_h = mlp(pi_h, prim_dict['layer']['policy'], self.activ_fn, layer_norm=self.layer_norm)
                    weight = tf.layers.dense(pi_h, len(prim_dict['act'][1]), activation='softmax')
                    weight = tf.clip_by_value(weight, weight_EPS, 1-weight_EPS)
                    self.weight[name] = weight

                    # NOTE: biased weight
                    if main_tail:
                        weight_biased = True
                        weight_bias = tf.layers.dense(pi_h, 1, activation='softmax')
                        weight_ones = tf.ones_like(weight_bias) * (1-weight_EPS)
                        weight_zeros = tf.zeros_like(weight_bias) * weight_EPS
                        weight_bias_0 = tf.concat([weight_ones,weight_zeros], axis=-1)
                        weight_bias_1 = tf.concat([weight_zeros,weight_ones], axis=-1)
                        weight_ph_0 = tf.ones_like(weight_bias) * self.weight_0_ph
                        weight_ph_1 = tf.ones_like(weight_bias) * self.weight_1_ph
                        weight_bias_ph = tf.concat([weight_ph_0, weight_ph_1], axis=-1)
                        self.weight_0[name] = weight_bias_0
                        self.weight_1[name] = weight_bias_1
                        self.weight_ph[name] = weight_bias_ph
                        self.weight_tf = weight
                        self.log_weight = tf.log(weight)
                    else:
                        self.weight_0[name] = weight
                        self.weight_1[name] = weight
                        self.weight_ph[name] = weight
                    
                    subgoal_dict = {}
                    if prim_dict.get('subgoal', None) is not None:
                        for prim_name, obs_idx in prim_dict['subgoal'].items():
                            assert prim_name in tails, "\n\t\033[91m[ERROR]: name of the target primitive not in tails or does not match\033[0m"
                            with tf.variable_scope('subgoal_'+prim_name, reuse=False):
                                # NOTE(tmmichi): restriction(bounds) on subgoal.
                                subgoal_obs = tf.layers.dense(pi_h, len(obs_idx), activation='tanh') * 0.3
                                self.subgoal[prim_name] = subgoal_obs
                                # subgoal index: in an increasing order of observation
                                subgoal_dict[prim_name] = [subgoal_obs, obs_idx]
                    print(name + " constructed")
                break
        else:
            raise NotImplementedError("\n\t\033[91m[ERROR]:Weight layer not in tail\033[0m")

        # Primitive setup
        for name in tails:
            prim_dict = primitives[name]
            print("main tail: ",prim_dict['main_tail'])
            print("current name: ", name)
            layer_name = prim_dict['layer_name'] if prim_dict['main_tail'] else name
            if subgoal_dict.get(name, None) is not None:
                # subgoal_dict from weight initialization
                subgoal_obs, subgoal_obs_idx = subgoal_dict[name]
                subgoal_layer = np.zeros([len(subgoal_obs_idx), obs.shape[1].value], dtype=np.float32)
                for i, idx in enumerate(subgoal_obs_idx):
                    subgoal_layer[i][idx] = 1
                subgoal_obs = tf.matmul(subgoal_obs, subgoal_layer)
                # NOTE(tmmichi): absolute subgoal -> relative subgoal
                new_obs = obs + subgoal_obs
            else:
                new_obs = obs

            if prim_dict['tails'] == None:
                if 'weight' not in name.split('/'):
                    print('\tGraph of ' + name + ' initializing...')
                    with tf.variable_scope(layer_name, reuse=reuse):
                        if self.feature_extraction == "cnn":
                            pi_h = self.cnn_extractor(new_obs, **self.cnn_kwargs)
                            raise NotImplementedError("Image input not supported for now")
                        else:
                            pi_h = tf.layers.flatten(new_obs)
                        
                        #------------- Input observation sieving layer -------------#
                        index_pair = {}
                        sieve_layer = np.zeros([new_obs.shape[1].value, len(prim_dict['obs'][1])], dtype=np.float32)
                        for i in range(len(prim_dict['obs'][1])):
                            index_pair[prim_dict['obs'][1][i]] = i
                            sieve_layer[prim_dict['obs'][1][i]][i] = 1
                        pi_h = tf.matmul(pi_h, sieve_layer)
                        #------------- Observation sieving layer End -------------#

                        if 'subtract' in prim_dict['obs_relativity'].keys():
                            print("\tIN SUBTRACT")
                            ref = prim_dict['obs_relativity']['subtract']['ref']
                            print('ref: ', ref)
                            tar = prim_dict['obs_relativity']['subtract']['tar']
                            assert len(ref) == len(tar), "Error: length of reference and target indicies unidentical"
                            ref_sieve = np.zeros([pi_h.shape[1].value, len(ref)], dtype=np.float32)
                            tar_sieve = np.zeros([pi_h.shape[1].value, len(tar)], dtype=np.float32)
                            remainder_list = list(range(pi_h.shape[1].value))
                            for i in range(len(ref)):
                                remainder_list.remove(index_pair[ref[i]])
                                remainder_list.remove(index_pair[tar[i]])
                                ref_sieve[index_pair[ref[i]]][i] = 1
                                tar_sieve[index_pair[tar[i]]][i] = 1
                            ref_obs = tf.matmul(pi_h, ref_sieve)
                            tar_obs = tf.matmul(pi_h, tar_sieve)
                            subs_obs = ref_obs - tar_obs
                            if not len(remainder_list) == 0:
                                remainder = pi_h.shape[1].value - (len(ref) + len(tar))
                                left_sieve = np.zeros([pi_h.shape[1].value, remainder], dtype=np.float32)
                                for j in range(len(remainder_list)):
                                    left_sieve[remainder_list[j]][j] = 1
                                rem_obs = tf.matmul(pi_h, left_sieve)
                                pi_h = tf.concat([subs_obs, rem_obs], axis=-1)
                            else:
                                pi_h = subs_obs

                        pi_h = mlp(pi_h, prim_dict['layer']['policy'], self.activ_fn, layer_norm=self.layer_norm)

                        # if aux, then the produced action becomes scaled
                        # TODO(tmmichi): constant action scale with 0.1 does not represent same effect after being squashed
                        mu_ = tf.layers.dense(pi_h, len(prim_dict['act'][1]), activation=None) * prim_dict.get('act_scale',1)                        

                        if not non_log:
                            log_std = tf.layers.dense(pi_h, len(prim_dict['act'][1]), activation=None)
                        else:
                            std = tf.layers.dense(pi_h, len(prim_dict['act'][1]), activation='relu') + EPS
                            log_std = tf.log(std)

                        mu_array.append(mu_)
                        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)
                        log_std_array.append(log_std)
                        act_index.append(prim_dict['act'][1])
                        
                        self.primitive_actions[name] = mu_
                        self.primitive_log_std[name] = log_std

                        tf.summary.merge_all()
                    print('\t' + name + " constructed")
            else:
                if 'weight' not in name.split('/'):
                    with tf.variable_scope(layer_name, reuse=reuse):
                        _, mu_, log_std_ = self.construct_actor_graph(new_obs, primitives, prim_dict['tails'], total_action_dimension, reuse, non_log)
                    mu_array.append(mu_)
                    log_std_array.append(log_std_)
                    act_index.append(prim_dict['act'][1])
                    print('\t' + name + " constructed")
        
        assert not isinstance(weight, type(None)), \
            '\n\t\033[91m[ERROR]: No weight within tail:{0}\033[0m'.format(tails)
        selected_idx = None
        pi_MCP, mu_MCP, log_std_MCP = fuse_networks_MCP(mu_array, log_std_array, weight, act_index, total_action_dimension)
        # pi_MCP, mu_MCP, log_std_MCP, selected_idx = fuse_networks_categorical(mu_array, log_std_array, weight, act_index, total_action_dimension)
        self.selected_idx = selected_idx
        if weight_biased:
            pi_0, mu_0, log_std_0 = fuse_networks_MCP(mu_array, log_std_array, weight_bias_0, act_index, total_action_dimension)
            pi_1, mu_1, log_std_1 = fuse_networks_MCP(mu_array, log_std_array, weight_bias_1, act_index, total_action_dimension)
            pi_ph, mu_ph, log_std_ph = fuse_networks_MCP(mu_array, log_std_array, weight_bias_ph, act_index, total_action_dimension)
            return pi_MCP, mu_MCP, log_std_MCP, [pi_0, mu_0, log_std_0], [pi_1, mu_1, log_std_1], [pi_ph, mu_ph, log_std_ph]
        else:
            return pi_MCP, mu_MCP, log_std_MCP
  
    def construct_value_graph(self, obs=None, action=None, primitives=None, tails=None, reuse=False, 
                                create_vf=False, create_qf=False, qf1=False, qf2=False, weight=False, SACD=True):
        print("Received tails in value graph: ",tails)
        if obs is None:
            obs = self.processed_obs

        for name in tails:
            # print('name: ', name)
            prim_dict = primitives[name]
            if prim_dict['load_value']:
                # print('in load_value')
                layer_name = prim_dict['layer_name'] if prim_dict['main_tail'] else name
                if prim_dict['tails'] == None:
                    with tf.variable_scope(layer_name, reuse=reuse):
                        if self.feature_extraction == "cnn":
                            critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
                            raise NotImplementedError("Image input not supported for now")
                        else:
                            critics_h = tf.layers.flatten(obs)

                        #------------- Input observation sieving layer -------------#
                        sieve_layer = np.zeros([obs.shape[1].value, len(prim_dict['obs'][1])], dtype=np.float32)
                        for i in range(len(prim_dict['obs'][1])):
                            sieve_layer[prim_dict['obs'][1][i]][i] = 1
                        critics_h = tf.matmul(critics_h, sieve_layer)
                        #------------- Observation sieving layer End -------------#

                        if create_vf:
                            # Value function
                            vf_h = mlp(critics_h, prim_dict['layer']['value'], self.activ_fn, layer_norm=self.layer_norm)
                            value_fn = tf.layers.dense(vf_h, 1, name="vf")
                            self.value_fn += value_fn
                            self.primitive_value[layer_name] = value_fn

                        if create_qf:
                            if not weight:
                                #------------- Input action sieving layer -------------#
                                sieve_layer = np.zeros([action.shape[1], len(prim_dict['act'][1])], dtype=np.float32)
                                for i in range(len(prim_dict['act'][1])):
                                    sieve_layer[prim_dict['act'][1][i]][i] = 1
                                qf_h = tf.matmul(action, sieve_layer)
                                #------------- Action sieving layer End -------------#
                            else:
                                qf_h = action
                            
                            # Concatenate preprocessed state and action
                            qf_h = tf.concat([critics_h, qf_h], axis=-1)

                            qf_h = mlp(qf_h, prim_dict['layer']['value'], self.activ_fn, layer_norm=self.layer_norm)
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
                        self.construct_value_graph(obs, action, primitives, prim_dict['tails'], reuse, create_vf, create_qf, qf1, qf2)
            if 'weight' in name.split('/'):
                # print('in weight')
                composite_name = name.split('/')[0]
                layer_name = 'train/'+composite_name if prim_dict['main_tail'] else composite_name
                with tf.variable_scope(layer_name, reuse=reuse):
                    if self.feature_extraction == "cnn":
                        critics_h = self.cnn_extractor(obs, **self.cnn_kwargs)
                        raise NotImplementedError("Image input not supported for now")
                    else:
                        critics_h = tf.layers.flatten(obs)

                    #------------- Input observation sieving layer -------------#
                    sieve_layer = np.zeros([obs.shape[1].value, len(prim_dict['obs'][1])], dtype=np.float32)
                    for i in range(len(prim_dict['obs'][1])):
                        sieve_layer[prim_dict['obs'][1][i]][i] = 1
                    critics_h = tf.matmul(critics_h, sieve_layer)
                    #------------- Observation sieving layer End -------------#

                    if create_vf:
                        # Value function
                        vf_h = mlp(critics_h, prim_dict['layer']['value'], self.activ_fn, layer_norm=self.layer_norm)
                        value_fn = tf.layers.dense(vf_h, 1, name="vf")
                        self.value_fn += value_fn
                        self.primitive_value[layer_name] = value_fn

                    if create_qf:
                        if not weight:
                            #------------- Input action sieving layer -------------#
                            sieve_layer = np.zeros([action.shape[1], len(prim_dict['composite_action_index'])], dtype=np.float32)
                            for i in range(len(prim_dict['composite_action_index'])):
                                sieve_layer[prim_dict['composite_action_index']][i] = 1
                            qf_h = tf.matmul(action, sieve_layer)
                            #------------- Action sieving layer End -------------#
                        else:
                            qf_h = None if SACD else action
                        
                        # Concatenate preprocessed state and action
                        qf_h = critics_h if SACD else tf.concat([critics_h, qf_h], axis=-1)
                        qf_h = mlp(qf_h, prim_dict['layer']['value'], self.activ_fn, layer_norm=self.layer_norm)
                        weight_dim = len(prim_dict['act'][1]) if SACD else 1
                        if qf1:
                            qf = tf.layers.dense(qf_h, weight_dim, name="qf1")
                            self.qf1 += qf
                            self.primitive_qf1[layer_name] = qf
                        if qf2:
                            qf = tf.layers.dense(qf_h, weight_dim, name="qf2")
                            self.qf2 += qf
                            self.primitive_qf2[layer_name] = qf

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run(self.deterministic_policy, {self.obs_ph: obs})
        return self.sess.run(self.policy, {self.obs_ph: obs})
    
    def subgoal_step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_policy, self.subgoal, self.weight], {self.obs_ph: obs})
        return self.sess.run([self.policy, self.subgoal, self.weight], {self.obs_ph: obs})
    
    def subgoal_step_temp(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_policy, self.subgoal, self.weight, self.selected_idx], {self.obs_ph: obs})
        return self.sess.run([self.policy, self.subgoal, self.weight, self.selected_idx], {self.obs_ph: obs})
    
    def biased_subgoal_step(self, obs, index):
        if index == 0:
            return self.sess.run([self.policy_0, self.subgoal, self.weight_0], {self.obs_ph: obs})
        elif index == 1:
            return self.sess.run([self.policy_1, self.subgoal, self.weight_1], {self.obs_ph: obs})
        elif type(index) == list:
            return self.sess.run([self.policy_ph, self.subgoal, self.weight_ph], \
                {self.obs_ph: obs, self.weight_0_ph: index[0],self.weight_1_ph: index[1]})
    
    def get_weight(self, obs):
        return self.sess.run(self.weight, {self.obs_ph: obs})
    
    def get_tf_weight(self):
        return self.weight_tf
    
    def get_log_weight(self):
        return self.log_weight

    def get_sliced_tf_weight(self):
        return tf.slice(self.weight_tf, [0, 0], [-1,1])
    
    def get_sliced_log_weight(self):
        return tf.slice(self.log_weight, [0, 0], [-1,1])

    def get_primitive_action(self, obs):
        return self.sess.run(self.primitive_actions, {self.obs_ph: obs})

    def get_primitive_log_std(self, obs):
        return self.sess.run(self.primitive_log_std, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run([self.act_mu, self.std], {self.obs_ph: obs})
    
    def kl(self, other):
        return self.dist.kl_divergence(other.dist)


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



register_policy("CnnPolicy_sac", CnnPolicy)
register_policy("LnCnnPolicy_sac", LnCnnPolicy)
register_policy("MlpPolicy_sac", MlpPolicy)
register_policy("LnMlpPolicy_sac", LnMlpPolicy)
