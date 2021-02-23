from abc import abstractmethod

import tensorflow as tf
import numpy as np

from stable_baselines.common.policies import ActorCriticPolicy, nature_cnn, register_policy
from stable_baselines.common.tf_layers import mlp

EPS = 1e-6  # Avoid NaN (prevents division by zero or log of zero)
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
    # task_list = ['aux','reach','grasp']
    task_list = ['reach','grasp']
    with tf.variable_scope("fuse"):
        mu_temp = std_sum = tf.tile(tf.reshape(weight[:,0],[-1,1]), tf.constant([1,total_action_dimension])) * 0
        for i in range(len(mu_array)):
            weight_cutoff = tf.clip_by_value(weight[:,i], EPS, 1-EPS)
            tf.summary.histogram('weight '+task_list[i], tf.reshape(weight_cutoff,[-1,1]))
            weight_tile = tf.tile(tf.reshape(weight_cutoff,[-1,1]), tf.constant([1,mu_array[i][0].shape[0].value]))
            normed_weight_index = tf.math.divide_no_nan(weight_tile, tf.exp(log_std_array[i]))
            mu_weighted_i = mu_array[i] * normed_weight_index
            shaper = np.zeros([len(act_index[i]), total_action_dimension], dtype=np.float32)
            for j, index in enumerate(act_index[i]):
                shaper[j][index] = 1
            mu_temp += tf.matmul(mu_weighted_i, shaper)
            std_sum += tf.matmul(normed_weight_index, shaper)
        std_MCP = tf.math.reciprocal_no_nan(std_sum)
        mu_MCP = tf.math.multiply(mu_temp, std_MCP, name="mu_MCP")
        # log_std_MCP = tf.log(tf.clip_by_value(std_MCP, LOG_STD_MIN, LOG_STD_MAX), name="log_std_MCP")
        log_std_MCP = tf.log(std_MCP, name="log_std_MCP")
        pi_MCP = tf.math.add(mu_MCP, tf.random_normal(tf.shape(mu_MCP)) * tf.exp(log_std_MCP), name="pi_MCP")
        # pi_MCP = mu_MCP
    
    return pi_MCP, mu_MCP, log_std_MCP


class HPCPPOPolicy(ActorCriticPolicy):
    """
    Policy object that implements a HPCPPOPolicy.

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
        super(HPCPPOPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
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
        self.subgoal = {}
        self.primitive_actions = {}
        self.primitive_log_std = {}
        self.primitive_value = {}
        self.primitive_vf = {}
        self.primitive_qf = {}
        self.reg_loss = None
        self.reg_weight = reg_weight
        self.activ_fn = act_fun

    def make_HPC_actor(self, obs=None, primitives=None, tails=None, total_action_dimension=0, reuse=False, scope="pi"):
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

        if obs is None:
            obs = self.processed_obs
        with tf.variable_scope(scope, reuse=reuse):
            pi_MCP, mu_MCP, log_std_MCP = self.construct_actor_graph(obs, primitives, tails, total_action_dimension, reuse, non_log)

        logp_pi = gaussian_likelihood(pi_MCP, mu_MCP, log_std_MCP)
        self.entropy = gaussian_entropy(log_std_MCP)
        self.std = tf.exp(log_std_MCP)
        
        # policies with squashing func at test time
        deterministic_policy, policy, logp_pi = apply_squashing_func(mu_MCP, pi_MCP, logp_pi)
        weight_val = [self.weight[name] for name in self.weight.keys()]
        policy = tf.Print(policy,[mu_MCP, self.std, pi_MCP, logp_pi, weight_val], "mu, std, pi, logpi, weight: ", summarize=-1)
        self.neglogp = -logp_pi
        self.policy = policy
        self.deterministic_policy = deterministic_policy

        return deterministic_policy, policy, logp_pi
 
    def make_HPC_critics(self, obs=None, action=None, primitives=None, tails=None, scope="values_fn", reuse=False):
        """
        Creates the two Q-Values approximator along with the HPC Value function

        :param primitives: (dict) Obs/act information of primitives
        :param obs: (TensorFlow Tensor) The observation placeholder (can be None for default placeholder)
        :param action: (TensorFlow Tensor) The action placeholder
        :param reuse: (bool) whether or not to reuse parameters
        :param scope: (str) the scope name
        :return: ([tf.Tensor]) Mean, action and log probability
        """
        self.qf = 0
        self.value_fn = 0

        if obs is None:
            obs = self.processed_obs

        with tf.variable_scope(scope, reuse=reuse):
            with tf.variable_scope('qf', reuse=reuse):
                self.construct_value_graph(obs, action, primitives, tails, reuse=reuse, qf=True)
            with tf.variable_scope('vf', reuse=reuse):
                self.construct_value_graph(obs, action, primitives, tails, reuse=reuse, vf=True)

        return self.qf, self.value_fn

    def construct_actor_graph(self, obs=None, primitives=None, tails=None, total_action_dimension=0, reuse=False, non_log=False):
        print("Received tails in actor graph: ",tails)
        if obs is None:
            obs = self.processed_obs

        mu_array = []
        log_std_array = []
        act_index = []
        weight = None

        # Meta-controller Initialization
        for name in tails:
            if 'weight' in name.split('/'):
                print('Graph of '+name+' initializing')
                prim_dict = primitives[name]
                layer_name = prim_dict['layer_name'] if prim_dict['main_tail'] else name
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
                    self.weight[name] = weight
                    
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
        pi_MCP, mu_MCP, log_std_MCP = fuse_networks_MCP(mu_array, log_std_array, weight, act_index, total_action_dimension)
        
        return pi_MCP, mu_MCP, log_std_MCP
  
    def construct_value_graph(self, obs=None, action=None, primitives=None, tails=None, reuse=False, vf=False, qf=False):
        print("Received tails in value graph: ",tails)
        if obs is None:
            obs = self.processed_obs

        for name in tails:
            prim_dict = primitives[name]
            if prim_dict['load_value']:
                print('value loaded')
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

                        #------------- Input action sieving layer -------------#
                        sieve_layer = np.zeros([action.shape[1], len(prim_dict['act'][1])], dtype=np.float32)
                        for i in range(len(prim_dict['act'][1])):
                            sieve_layer[prim_dict['act'][1][i]][i] = 1
                        qf_h = tf.matmul(action, sieve_layer)
                        #------------- Action sieving layer End -------------#
                        
                        # Concatenate preprocessed state and action
                        qf_h = tf.concat([critics_h, qf_h], axis=-1)

                        qf_h = mlp(qf_h, prim_dict['layer']['value'], self.activ_fn, layer_norm=self.layer_norm)
                        qf = tf.layers.dense(qf_h, 1, name="qf")
                        self.qf += qf
                        self.primitive_[layer_name] = qf
                else:
                    with tf.variable_scope(layer_name, reuse=reuse):
                        self.construct_value_graph(obs, action, primitives, prim_dict['tails'], reuse)
            if 'weight' in name.split('/'):
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

                    if vf:
                        vf_h = mlp(vf_h, prim_dict['layer']['value'], self.activ_fn, layer_norm=self.layer_norm)
                        vf = tf.layers.dense(vf_h, 1, name='vf')
                        self.value_fn += vf
                        self.primitive_vf[layer_name] = vf

                    if qf:
                        #------------- Input action sieving layer -------------#
                        sieve_layer = np.zeros([action.shape[1], len(prim_dict['composite_action_index'])], dtype=np.float32)
                        for i in range(len(prim_dict['composite_action_index'])):
                            sieve_layer[prim_dict['composite_action_index']][i] = 1
                        qf_h = tf.matmul(action, sieve_layer)
                        #------------- Action sieving layer End -------------#
                        
                        # Concatenate preprocessed state and action
                        qf_h = tf.concat([critics_h, qf_h], axis=-1)
                        qf_h = mlp(qf_h, prim_dict['layer']['value'], self.activ_fn, layer_norm=self.layer_norm)
                        qf = tf.layers.dense(qf_h, 1, name="qf")
                        self.qf += qf
                        self.primitive_qf[layer_name] = qf

    def step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run(self.deterministic_policy, {self.obs_ph: obs})
        return self.sess.run(self.policy, {self.obs_ph: obs})
    
    def value(self, obs, state=None, mask=None):
        return self.sess.run(self.value_flat, {self.obs_ph: obs})
    
    def subgoal_step(self, obs, state=None, mask=None, deterministic=False):
        if deterministic:
            return self.sess.run([self.deterministic_policy, self.subgoal, self.weight], {self.obs_ph: obs})
        return self.sess.run([self.policy, self.subgoal, self.weight], {self.obs_ph: obs})
    
    def get_weight(self, obs):
        return self.sess.run(self.weight, {self.obs_ph: obs})

    def get_primitive_action(self, obs):
        return self.sess.run(self.primitive_actions, {self.obs_ph: obs})

    def get_primitive_log_std(self, obs):
        return self.sess.run(self.primitive_log_std, {self.obs_ph: obs})

    def proba_step(self, obs, state=None, mask=None):
        return self.sess.run([self.act_mu, self.std], {self.obs_ph: obs})
    
    def kl(self, other):
        return self.dist.kl_divergence(other.dist)
    
    def get_logstd(self, obs):
        obs = np.array(obs)
        obs = obs.reshape((-1,) + obs.shape)
        logstd = tf.log(self.std)
        return self.sess.run([logstd], {self.obs_ph: obs})


class CnnPolicy(HPCPPOPolicy):
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

class LnCnnPolicy(HPCPPOPolicy):
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

class MlpPolicy(HPCPPOPolicy):
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

class LnMlpPolicy(HPCPPOPolicy):
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


register_policy("CnnPolicy_hpcppo", CnnPolicy)
register_policy("LnCnnPolicy_hpcppo", LnCnnPolicy)
register_policy("MlpPolicy_hpcppo", MlpPolicy)
register_policy("LnMlpPolicy_hpcppo", LnMlpPolicy)
