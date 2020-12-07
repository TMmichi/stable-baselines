import os
import glob
import json
import zipfile
import warnings
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from typing import Union, List, Callable, Optional

import gym
import cloudpickle
import numpy as np
import tensorflow as tf

from stable_baselines.common.radam import RAdamOptimizer
from stable_baselines.common.misc_util import set_global_seeds
from stable_baselines.common.math_util import unscale_action
from stable_baselines.common.save_util import data_to_json, json_to_data, params_to_bytes, bytes_to_params
from stable_baselines.common.policies import get_policy_from_name, ActorCriticPolicy
from stable_baselines.common.runners import AbstractEnvRunner
from stable_baselines.common.vec_env import (VecEnvWrapper, VecEnv, DummyVecEnv,
                                             VecNormalize, unwrap_vec_normalize)
from stable_baselines.common.callbacks import BaseCallback, CallbackList, ConvertCallback
from stable_baselines import logger


class BaseRLModel(ABC):
    """
    The base RL model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    :param policy_base: (BasePolicy) the base policy used by this method
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    :param composite_primitive_name: (str) Name of the composite primitive. If None, will have no effect.
    """

    def __init__(self, policy, env, verbose=0, *, requires_vec_env, policy_base,
                 policy_kwargs=None, seed=None, n_cpu_tf_sess=None, composite_primitive_name=None):
        if isinstance(policy, str) and policy_base is not None:
            self.policy = get_policy_from_name(policy_base, policy)
        else:
            self.policy = policy
        self.env = env
        self.verbose = verbose
        self._requires_vec_env = requires_vec_env
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs
        self.tails = []
        self.observation_space = None
        self.action_space = None
        self.n_envs = None
        self._vectorize_action = False
        self.num_timesteps = 0
        self.graph = None
        self.sess = None
        self.params = None
        self.seed = seed
        self._param_load_ops = None
        self.n_cpu_tf_sess = n_cpu_tf_sess
        self.episode_reward = None
        self.ep_info_buf = None
        self.composite_primitive_name = composite_primitive_name
        self.primitives = OrderedDict()
        self.top_hierarchy_level = 0

        if env is not None:
            print("env is not none")
            if isinstance(env, str):
                if self.verbose >= 1:
                    print("Creating environment from the given name, wrapped in a DummyVecEnv.")
                self.env = env = DummyVecEnv([lambda: gym.make(env)])

            self.observation_space = env.observation_space
            self.action_space = env.action_space
            if requires_vec_env:
                if isinstance(env, VecEnv):
                    self.n_envs = env.num_envs
                else:
                    # The model requires a VecEnv
                    # wrap it in a DummyVecEnv to avoid error
                    self.env = DummyVecEnv([lambda: env])
                    if self.verbose >= 1:
                        print("Wrapping the env in a DummyVecEnv.")
                    self.n_envs = 1
            else:
                if isinstance(env, VecEnv):
                    if env.num_envs == 1:
                        self.env = _UnvecWrapper(env)
                        self._vectorize_action = True
                    else:
                        raise ValueError("Error: the model requires a non vectorized environment or a single vectorized"
                                         " environment.")
                self.n_envs = 1

        # Get VecNormalize object if it exists
        self._vec_normalize_env = unwrap_vec_normalize(self.env)

    def get_env(self):
        """
        returns the current environment (can be None if not defined)

        :return: (Gym Environment) The current environment
        """
        return self.env

    def get_vec_normalize_env(self) -> Optional[VecNormalize]:
        """
        Return the ``VecNormalize`` wrapper of the training env
        if it exists.

        :return: Optional[VecNormalize] The ``VecNormalize`` env.
        """
        return self._vec_normalize_env

    def set_env(self, env):
        """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.

        :param env: (Gym Environment) The environment for learning a policy
        """
        if env is None and self.env is None:
            if self.verbose >= 1:
                print("Loading a model without an environment, "
                      "this model cannot be trained until it has a valid environment.")
            return
        elif env is None:
            raise ValueError("Error: trying to replace the current environment with None")

        # sanity checking the environment
        assert self.observation_space == env.observation_space, \
            "Error: the environment  passed must have at least the same observation space as the model was trained on. self.obs = {0}, env.obs = {1}".format(self.observation_space, env.observation_space)
        assert self.action_space == env.action_space, \
            "Error: the environment passed must have at least the same action space as the model was trained on. self.act = {0}, env.act = {1}".format(self.action_space, env.action_space)

        if self._requires_vec_env:
            assert isinstance(env, VecEnv), \
                "Error: the environment passed is not a vectorized environment, however {} requires it".format(
                    self.__class__.__name__)
            assert not self.policy.recurrent or self.n_envs == env.num_envs, \
                "Error: the environment passed must have the same number of environments as the model was trained on." \
                "This is due to the Lstm policy not being capable of changing the number of environments."
            self.n_envs = env.num_envs
        else:
            # for models that dont want vectorized environment, check if they make sense and adapt them.
            # Otherwise tell the user about this issue
            if isinstance(env, VecEnv):
                if env.num_envs == 1:
                    env = _UnvecWrapper(env)
                    self._vectorize_action = True
                else:
                    raise ValueError("Error: the model requires a non vectorized environment or a single vectorized "
                                     "environment.")
            else:
                self._vectorize_action = False

            self.n_envs = 1

        self.env = env
        self._vec_normalize_env = unwrap_vec_normalize(env)

        # Invalidated by environment change.
        self.episode_reward = None
        self.ep_info_buf = None

    def _init_num_timesteps(self, reset_num_timesteps=True):
        """
        Initialize and resets num_timesteps (total timesteps since beginning of training)
        if needed. Mainly used logging and plotting (tensorboard).

        :param reset_num_timesteps: (bool) Set it to false when continuing training
            to not create new plotting curves in tensorboard.
        :return: (bool) Whether a new tensorboard log needs to be created
        """
        if reset_num_timesteps:
            self.num_timesteps = 0

        new_tb_log = self.num_timesteps == 0
        return new_tb_log

    @abstractmethod
    def setup_model(self):
        """
        Create all the functions and tensorflow graphs necessary to train the model
        """
        pass

    def _init_callback(self,
                      callback: Union[None, Callable, List[BaseCallback], BaseCallback]
                      ) -> BaseCallback:
        """
        :param callback: (Union[None, Callable, List[BaseCallback], BaseCallback])
        :return: (BaseCallback)
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)
        # Convert functional callback to object
        if not isinstance(callback, BaseCallback):
            callback = ConvertCallback(callback)

        callback.init_callback(self)
        return callback

    def set_random_seed(self, seed: Optional[int]) -> None:
        """
        :param seed: (Optional[int]) Seed for the pseudo-random generators. If None,
            do not change the seeds.
        """
        # Ignore if the seed is None
        if seed is None:
            return
        # Seed python, numpy and tf random generator
        set_global_seeds(seed)
        if self.env is not None:
            self.env.seed(seed)
            # Seed the action space
            # useful when selecting random actions
            self.env.action_space.seed(seed)
        self.action_space.seed(seed)

    def _setup_learn(self):
        """
        Check the environment.
        """
        if self.env is None:
            raise ValueError("Error: cannot train the model without a valid environment, please set an environment with"
                             "set_env(self, env) method.")
        if self.episode_reward is None:
            self.episode_reward = np.zeros((self.n_envs,))
        if self.ep_info_buf is None:
            self.ep_info_buf = deque(maxlen=100)

    def construct_primitive_info(self, name, freeze, level, obs_range: Union[dict, int], obs_index, act_range: Union[dict, int], act_index, act_scale, obs_relativity, layer_structure, subgoal=None, loaded_policy=None, load_value=True):
        '''
        Returns info of the primitive as a dictionary

        :param name: (str) name of the primitive
        :param freeze: (bool) primitive to be frozen at training time
        :param level: (int) hierarchical level of the primitive
        :param obs_range: (dict or int) a dictionary containing min/max bound of the observation range. If int, then range fixed to 0
        :param obs_index: ([int, ...]) a list of indices of the observation for the primitive
        :param act_range: (dict or int) a dictionary containing min/max bound of the action range. If int, then all dimensions of the range fixed to [0,1]
        :param act_index: ([int, ...]) a list of indices of the action for the primitive
        :param act_scale: (int) action scale
        :param obs_relativity: (dict) a dictionary of observation index-relativity
        :param layer_structure: (dict) layer structure of the primitive policy/value
        :param subgoal: (dict) a dictionary containing [primitive name, subgoal->obs index]
        :param loaded_policy: ((dict, dict)) tuple of data and parameters for pretrained policy.zip
        :param load_value: (bool) load separate value network for the primitive
        :return: (dict: {'obs':tuple, 'act':tuple, 'layer':dict}) primitive information
        '''
        if isinstance(loaded_policy, type(None)):
            assert not freeze, \
                '\n\t\033[91m[ERROR]: Newly appointed primitive at training time cannot be frozen\033[0m'
            assert name != None, \
                '\n\t\033[91m[ERROR]: Newly appointed primitive should have its name designated\033[0m'
            if name == 'weight':
                self.top_hierarchy_level = level
                name = self.composite_primitive_name + "/" + name
            layer_name = '/'.join(['train','level'+str(level)+'_'+name])
            primitive_name = '/'.join(['level'+str(level)+'_'+name])
            self.tails.append(primitive_name)

            obs_dimension = len(obs_index)
            if isinstance(obs_range, dict):
                assert 'min' in obs_range.keys() and 'max' in obs_range.keys(), \
                    '\n\t\033[91m[ERROR]: No keys named "min" or "max" within the obs_range.\033[0m'
                assert len(obs_range['min']) == len(obs_range['max']), \
                    '\n\t\033[91m[ERROR]: Length of minimum obs_range and maximum obs_range differs.\033[0m'
                assert len(obs_range['min']) == obs_dimension, \
                    '\n\t\033[91m[ERROR]: length of obs_range differs with the length of obs_index.\033[0m'
                obs_range_max = np.array(obs_range['max'])
                obs_range_min = np.array(obs_range['min'])
            elif isinstance(obs_range, int):
                obs_range_max = np.array([0]*obs_dimension)
                obs_range_min = np.array([0]*obs_dimension)
            else:
                raise TypeError("\n\t\033[91m[ERROR]: obs_range wrong type - Should be a dict or an int. Received {0}\033[0m".format(type(obs_range)))
            obs_index.sort()
            obs_space = gym.spaces.Box(obs_range_min, obs_range_max, dtype=np.float32)
            obs = (obs_space, obs_index)

            act_dimension = len(act_index)
            if isinstance(act_range, dict):
                assert 'min' in act_range.keys() and 'max' in act_range.keys(), \
                    '\n\t\033[91m[ERROR]: No keys named "min" or "max" within the act_range.\033[0m'
                assert len(act_range['min']) == len(act_range['max']), \
                    '\n\t\033[91m[ERROR]: Length of the minimum act_range and the maximum act_range differs.\033[0m'
                assert len(act_range['min']) == act_dimension, \
                    '\n\t\033[91m[ERROR]: Length of the act_range differs with the length of act_index.\033[0m'
                act_range_max = np.array(act_range['max'])
                act_range_min = np.array(act_range['min'])
            elif isinstance(act_range, int):
                act_range_max = np.array([1]*act_dimension)
                act_range_min = np.array([0]*act_dimension)
            else:
                raise TypeError("\n\t\033[91m[ERROR]: act_index wrong type, should be a dict or an int. Received {0}\033[0m".format(type(act_index)))
            act_index.sort()
            act_space = gym.spaces.Box(act_range_min, act_range_max, dtype=np.float32)
            act = (act_space, act_index)

            assert (name == 'weight') and subgoal is not None, "Error: Only weight primitive can have subgoal argument"
            policy_layer_structure = layer_structure['policy']
            value_layer_structure = layer_structure.get('value',None)
            tails = None
            main_tail = True
            load_value = False
            
        elif isinstance(loaded_policy, tuple):
            data_dict, param_dict = loaded_policy
            submodule_primitive = data_dict.get('primitives',OrderedDict())
            self.primitives = {**self.primitives, **submodule_primitive}

            if name is not None:
                loaded_name = data_dict.get('composite_primitive_name', None)
                if name != loaded_name and loaded_name is not None:
                    print("\n\t\033[93m[WARNING]: Name of the loaded policy ({0}) is different from the received name ({1}). {0} will be used.\033[0m".format(loaded_name,name))
                    name = loaded_name
                layer_name_list = ['freeze' if freeze else 'train','loaded','level'+str(level)+'_'+name]
                primitive_name_list = ['level'+str(level)+'_'+name]
                if level == 1:
                    layer_name_list.append('level0')
                    primitive_name_list.append('level0')
                    tails = None
                else:
                    tails = data_dict.get('tails',None)
                layer_name = '/'.join(layer_name_list)
                primitive_name = '/'.join(primitive_name_list)
                self.tails.append(primitive_name)
                main_tail = True
            else:
                primitive_name = 'loaded'
                layer_name = 'loaded'
                tails = data_dict['tails']
                self.tails = tails
                main_tail = True

            obs_space = data_dict['observation_space']
            act_space = data_dict['action_space']
            assert len(obs_index) == obs_space.shape[0], \
                '\n\t\033[91m[ERROR]: Loaded observation dimension mismatches with the length of obs_index. Loaded obs_dimension = {0}, len(obs_index) = {1}\033[0m'.format(obs_space.shape[0], len(obs_index))
            assert len(act_index) == act_space.shape[0], \
                '\n\t\033[91m[ERROR]: Loaded action dimension mismatches with the length of act_index. Loaded act_dimension = {0}, len(act_index) = {1}\033[0m'.format(act_space.shape[0], len(act_index))
            obs_index.sort()
            act_index.sort()

            obs = (obs_space, obs_index)
            act = (act_space, act_index)

            if 'pretrained_param' not in self.primitives.keys():
                self.primitives['pretrained_param'] = [[],{}]
            updated_layer_name, updated_param_dict = self.loaded_policy_name_update(layer_name, param_dict, load_value)
            self.primitives['pretrained_param'][0] += updated_layer_name
            self.primitives['pretrained_param'][1] = {**self.primitives['pretrained_param'][1], **updated_param_dict}
            policy_layer_structure, value_layer_structure = self.get_layer_structure((obs, act), param_dict, load_value)
        else:
            raise TypeError("\n\t\033[91m[ERROR]: loaded_policy wrong type - Should be None or a tuple. Received {0}\033[0m".format(type(loaded_policy)))
        
        self.primitives[primitive_name] = {'obs': obs, 'act': act, 'act_scale': act_scale, 'obs_relativity': obs_relativity, 'subgoal': subgoal, 'layer': {'policy': policy_layer_structure, 'value': value_layer_structure}, 'layer_name': layer_name, 'tails':tails, 'main_tail':main_tail, 'load_value': load_value}
        
    @staticmethod
    def loaded_policy_name_update(layer_name, loaded_policy_dict, load_value):
        '''
        Concatenate name of each layers with name of the primitive

        :param name: (str) name of the primitive layer
        :param loaded_policy_dict: (dict) Dictionary of parameters of layers by name
        :param load_value: (bool) Use loaded separate value network
        :return: (list, dict) List consisting names of layers with specified primitive
                              Dictionary of parameters by updated names
        '''
        layer_name_list = []
        layer_param_dict = {}
        for name, value in loaded_policy_dict.items():
            add_value = False
            name_elem = name.split("/")
            assert 'LayerNorm' not in name_elem, \
                "\n\t\033[91m[ERROR]: LayerNormalized policy is not supported for now. Try to load primitives with unnormalized layers\033[0m"

            if 'pi' in name_elem:
                insert_index = 2
                add_value = True
            elif 'values_fn' in name_elem and layer_name != 'loaded':
                if load_value:
                    insert_index = 3
                    add_value = True

            if add_value:
                if layer_name:
                    name_elem.insert(insert_index, layer_name)
                updated_name = '/'.join(name_elem)
                #print("Updated name: ",updated_name)
                layer_name_list.append(updated_name)
                layer_param_dict[updated_name] = value

        return layer_name_list, layer_param_dict

    @staticmethod
    def get_layer_structure(argument_tuple, loaded_policy_dict, load_value):
        # TODO: Change target/values_fn ~~
        '''
        Return layer structure of the policy/value from parameter dictionary

        :param argument_tuple: ((tuple, tuple)) Tuple containing info of observation and action
        :param loaded_policy_dict: (dict) Dictionary of parameters of layers by name
        :param load_value: (bool) Use loaded separate value network
        :return: (list, list) Layer structure of the policy/value
        '''
        obs, _ = argument_tuple
        obs_dim = len(obs[1])
        policy_layer_structure = []
        value_layer_structure = []
        # if loaded for pretraining: model/pi/fc0/kernel:0
        # if loaded for testing: model/pi/name_of_primitive/fc0/kernel:0
        for name, value in loaded_policy_dict.items():
            if name.find("pi/fc") > -1:
                if name.find("fc0/kernel") > -1:
                    assert obs_dim == value.shape[0], \
                        "\n\t\033[91m[ERROR/Loaded Primitive]: Observation input of param shape does not match with the observation box. Potential corruption occured\033[0m"
                if name.find("bias") > -1:
                    policy_layer_structure.append(value.shape[0])
            if load_value:
                if name.find('model/values_fn/vf/fc') > -1:
                    if name.find('bias') > -1:
                        value_layer_structure.append(value.shape[0])
        
        return policy_layer_structure, value_layer_structure

    @abstractmethod
    def get_parameter_list(self):
        """
        Get tensorflow Variables of model's parameters

        This includes all variables necessary for continuing training (saving / loading).

        :return: (list) List of tensorflow Variables
        """
        pass

    def get_parameters(self, hierarchical=False):
        """
        Get current model parameters as dictionary of variable name -> ndarray.

        :return: (OrderedDict) Dictionary of variable name -> ndarray of model's parameters.
        """
        parameters = self.get_parameter_list() #self.params
        parameter_values = self.sess.run(parameters)
        if not hierarchical:
            return_dictionary = OrderedDict((param.name, value) for param, value in zip(parameters, parameter_values))
        else:
            names = []
            for param in parameters:
                name_list = param.name.split('/')
                if 'train' in name_list:
                    name_list.remove('train')
                    try:
                        name_list.remove('loaded')
                    except Exception:
                        pass
                    name = "/".join(name_list)
                elif 'freeze' in name_list:
                    name_list.remove('freeze')
                    name_list.remove('loaded')
                    name = "/".join(name_list)
                else:
                    name = param.name
                names.append(name)
            return_dictionary = OrderedDict((name, value) for name, value in zip(names, parameter_values))

        return return_dictionary

    def _setup_load_operations(self):
        """
        Create tensorflow operations for loading model parameters
        """
        # Assume tensorflow graphs are static -> check
        # that we only call this function once
        if self._param_load_ops is not None:
            raise RuntimeError("Parameter load operations have already been created")
        # For each loadable parameter, create appropiate
        # placeholder and an assign op, and store them to
        # self.load_param_ops as dict of variable.name -> (placeholder, assign)
        loadable_parameters = self.get_parameter_list()

        # Use OrderedDict to store order for backwards compatibility with
        # list-based params
        self._param_load_ops = OrderedDict()
        with self.graph.as_default():
            for param in loadable_parameters:
                placeholder = tf.placeholder(dtype=param.dtype, shape=param.shape)
                # param.name is unique (tensorflow variables have unique names)
                self._param_load_ops[param.name] = (placeholder, param.assign(placeholder))

    @abstractmethod
    def _get_pretrain_placeholders(self):
        """
        Return the placeholders needed for the pretraining:
        - obs_ph: observation placeholder
        - actions_ph will be population with an action from the environment
            (from the expert dataset)
        - deterministic_actions_ph: e.g., in the case of a Gaussian policy,
            the mean.

        :return: ((tf.placeholder)) (obs_ph, actions_ph, deterministic_actions_ph)
        """
        pass

    def pretrain(self, dataset, n_epochs=10, learning_rate=1e-4,
                 adam_epsilon=1e-8, val_interval=None):
        """
        Pretrain a model using behavior cloning:
        supervised learning given an expert dataset.

        NOTE: only Box and Discrete spaces are supported for now.

        :param dataset: (ExpertDataset) Dataset manager
        :param n_epochs: (int) Number of iterations on the training set
        :param learning_rate: (float) Learning rate
        :param adam_epsilon: (float) the epsilon value for the adam optimizer
        :param val_interval: (int) Report training and validation losses every n epochs.
            By default, every 10th of the maximum number of epochs.
        :return: (BaseRLModel) the pretrained model
        """
        continuous_actions = isinstance(self.action_space, gym.spaces.Box)
        discrete_actions = isinstance(self.action_space, gym.spaces.Discrete)

        assert discrete_actions or continuous_actions, 'Only Discrete and Box action spaces are supported'

        # Validate the model every 10% of the total number of iteration
        if val_interval is None:
            # Prevent modulo by zero
            if n_epochs < 10:
                val_interval = 1
            else:
                val_interval = int(n_epochs / 10)

        with self.graph.as_default():
            with tf.variable_scope('pretrain'):
                if continuous_actions:
                    obs_ph, actions_ph, deterministic_actions_ph = self._get_pretrain_placeholders()
                    loss = tf.reduce_mean(tf.square(actions_ph - deterministic_actions_ph))
                else:
                    obs_ph, actions_ph, actions_logits_ph = self._get_pretrain_placeholders()
                    # actions_ph has a shape if (n_batch,), we reshape it to (n_batch, 1)
                    # so no additional changes is needed in the dataloader
                    actions_ph = tf.expand_dims(actions_ph, axis=1)
                    one_hot_actions = tf.one_hot(actions_ph, self.action_space.n)
                    loss = tf.nn.softmax_cross_entropy_with_logits_v2(
                        logits=actions_logits_ph,
                        labels=tf.stop_gradient(one_hot_actions)
                    )
                    loss = tf.reduce_mean(loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, epsilon=adam_epsilon)
                #optimizer = RAdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, weight_decay=0.0)
                optim_op = optimizer.minimize(loss, var_list=self.params)

            self.sess.run(tf.global_variables_initializer())

        if self.verbose > 0:
            print("Pretraining with Behavior Cloning...")

        for epoch_idx in range(int(n_epochs)):
            train_loss = 0.0
            # Full pass on the training set
            for _ in range(len(dataset.train_loader)):
                expert_obs, expert_actions = dataset.get_next_batch('train')
                feed_dict = {
                    obs_ph: expert_obs,
                    actions_ph: expert_actions,
                }
                train_loss_, _ = self.sess.run([loss, optim_op], feed_dict)
                train_loss += train_loss_

            train_loss /= len(dataset.train_loader)

            if self.verbose > 0 and (epoch_idx + 1) % val_interval == 0:
                val_loss = 0.0
                # Full pass on the validation set
                for _ in range(len(dataset.val_loader)):
                    expert_obs, expert_actions = dataset.get_next_batch('val')
                    val_loss_,  = self.sess.run([loss], {obs_ph: expert_obs,
                                                        actions_ph: expert_actions})
                    val_loss += val_loss_

                val_loss /= len(dataset.val_loader)
                if self.verbose > 0:
                    print("==== Training progress {:.2f}% ====".format(100 * (epoch_idx + 1) / n_epochs))
                    print('Epoch {}'.format(epoch_idx + 1))
                    print("Training loss: {:.6f}, Validation loss: {:.6f}".format(train_loss, val_loss))
                    print()
            # Free memory
            del expert_obs, expert_actions
        if self.verbose > 0:
            print("Pretraining done.")
        return self

    @abstractmethod
    def learn(self, total_timesteps, callback=None, log_interval=100, tb_log_name="run",
              reset_num_timesteps=True):
        """
        Return a trained model.

        :param total_timesteps: (int) The total number of samples to train on
        :param callback: (Union[callable, [callable], BaseCallback])
            function called at every steps with state of the algorithm.
            It takes the local and global variables. If it returns False, training is aborted.
            When the callback inherits from BaseCallback, you will have access
            to additional stages of the training (training start/end),
            please read the documentation for more details.
        :param log_interval: (int) The number of timesteps before logging.
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param reset_num_timesteps: (bool) whether or not to reset the current timestep number (used in logging)
        :return: (BaseRLModel) the trained model
        """
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
        pass

    @abstractmethod
    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        """
        If ``actions`` is ``None``, then get the model's action probability distribution from a given observation.

        Depending on the action space the output is:
            - Discrete: probability for each possible action
            - Box: mean and standard deviation of the action output

        However if ``actions`` is not ``None``, this function will return the probability that the given actions are
        taken with the given parameters (observation, state, ...) on this model. For discrete action spaces, it
        returns the probability mass; for continuous action spaces, the probability density. This is since the
        probability mass will always be zero in continuous spaces, see http://blog.christianperone.com/2019/01/
        for a good explanation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param actions: (np.ndarray) (OPTIONAL) For calculating the likelihood that the given actions are chosen by
            the model for each of the given parameters. Must have the same number of actions and observations.
            (set to None to return the complete action probability distribution)
        :param logp: (bool) (OPTIONAL) When specified with actions, returns probability in log-space.
            This has no effect if actions is None.
        :return: (np.ndarray) the model's (log) action probability
        """
        pass

    def load_parameters(self, load_path_or_dict, exact_match=True):
        """
        Load model parameters from a file or a dictionary

        Dictionary keys should be tensorflow variable names, which can be obtained
        with ``get_parameters`` function. If ``exact_match`` is True, dictionary
        should contain keys for all model's parameters, otherwise RunTimeError
        is raised. If False, only variables included in the dictionary will be updated.

        This does not load agent's hyper-parameters.

        .. warning::
            This function does not update trainer/optimizer variables (e.g. momentum).
            As such training after using this function may lead to less-than-optimal results.

        :param load_path_or_dict: (str or file-like or dict) Save parameter location
            or dict of parameters as variable.name -> ndarrays to be loaded.
        :param exact_match: (bool) If True, expects load dictionary to contain keys for
            all variables in the model. If False, loads parameters only for variables
            mentioned in the dictionary. Defaults to True.
        """
        # Make sure we have assign ops
        if self._param_load_ops is None:
            self._setup_load_operations()
        if isinstance(load_path_or_dict, dict):
            # Assume `load_path_or_dict` is dict of variable.name -> ndarrays we want to load
            params = load_path_or_dict
        elif isinstance(load_path_or_dict, list):
            warnings.warn("Loading model parameters from a list. This has been replaced " +
                          "with parameter dictionaries with variable names and parameters. " +
                          "If you are loading from a file, consider re-saving the file.",
                          DeprecationWarning)
            # Assume `load_path_or_dict` is list of ndarrays.
            # Create param dictionary assuming the parameters are in same order
            # as `get_parameter_list` returns them.
            params = dict()
            for i, param_name in enumerate(self._param_load_ops.keys()):
                params[param_name] = load_path_or_dict[i]
        else:
            # Assume a filepath or file-like.
            # Use existing deserializer to load the parameters.
            # We only need the parameters part of the file, so
            # only load that part.
            _, params = BaseRLModel._load_from_file(load_path_or_dict, load_data=False)
            params = dict(params)

        feed_dict = {}
        param_update_ops = []
        # Keep track of not-updated variables
        not_updated_variables = set(self._param_load_ops.keys())
        for param_name, param_value in params.items():
            try:
                placeholder, assign_op = self._param_load_ops[param_name]
                feed_dict[placeholder] = param_value
                # Create list of tf.assign operations for sess.run
                param_update_ops.append(assign_op)
                # Keep track which variables are updated
                not_updated_variables.remove(param_name)
                print("Loaded param: ",param_name)
            except Exception as e:
                print("Param not in graph: ",param_name)

        for param_name in not_updated_variables:
            print("Unloaded param: ", param_name)

        # Check that we updated all parameters if exact_match=True
        if exact_match and len(not_updated_variables) > 0:
            raise RuntimeError("Load dictionary did not contain all variables. " +
                               "Missing variables: {}".format(", ".join(not_updated_variables)))

        self.sess.run(param_update_ops, feed_dict=feed_dict)

    @abstractmethod
    def save(self, save_path, cloudpickle=False):
        """
        Save the current parameters to file

        :param save_path: (str or file-like) The save location
        :param cloudpickle: (bool) Use older cloudpickle format instead of zip-archives.
        """
        raise NotImplementedError()

    @classmethod
    @abstractmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Environment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        raise NotImplementedError()

    @staticmethod
    def pretrainer_load(model, policy, env, **kwargs):
        """
        Construct trainer from policy structure

        :param model: (Algorithm Class) class of an Algorithm which includes primitives dictionary as a member variable
        :param policy: (BasePolicy) Policy object
        :param primitives: (dict) primitives by name to which items assigned as an info of obs/act/layer_structure
        :param env: (Gym Environment) the new environment to run the loaded model on
        :param kwargs: extra arguments to change the model when loading
        """
        if 'loaded' not in model.primitives.keys():
            # Check the existence of 'train/weight' in primitives
            weight_name = model.weight_check(model.primitives, model.composite_primitive_name, model.top_hierarchy_level)
            
            # get total_obs_bound, total_act_bound
            ranges = model.range_primitive(model.primitives, model.composite_primitive_name, model.top_hierarchy_level)
            model.primitives[weight_name]['composite_action_index'] = list(range(len(ranges[1][0])))

            data = {'observation_space': gym.spaces.Box(ranges[0][0], ranges[0][1], dtype=np.float32), \
                    'action_space': gym.spaces.Box(ranges[1][0], ranges[1][1], dtype=np.float32)}
        else:
            data = {'observation_space': model.primitives['loaded']['obs'][0], \
                    'action_space': model.primitives['loaded']['act'][0]}
        
        model.__dict__.update(kwargs)
        model.__dict__.update(data)
        
        model.set_env(env)
        try:
            model.setup_custom_model(model.primitives)
        except Exception as e:
            print(e)
            raise NotImplementedError("\n\t\033[91m[ERROR]: Given algorithm does not support compository scheme. Try load(path) instead.\033[0m")

        model.load_parameters(model.primitives['pretrained_param'][1], exact_match=False)
        model.primitives.pop('pretrained_param')
        for tail in model.tails:
            model.primitives[tail]['main_tail'] = False
            
        return model
    
    @staticmethod
    def weight_check(primitives: dict, composite_primitive_name: str, level: int):
        # TODO: Change train/weight
        '''
        Check the existence of 'train/weight' in primitive dict

        :param primitives: (dict) obs/act/structure info of primitives
        '''
        weight_name = 'level'+str(level)+'_'+composite_primitive_name+"/weight"
        assert weight_name in primitives.keys(), \
            '\n\t\033[91m[ERROR]: No weight at the top level hierarchy. YOU MUST HAVE IT\033[0m'
        
        return weight_name
    
    @staticmethod
    def range_primitive(primitives: dict, composite_primitive_name: str, level: int) -> list:
        # TODO: Change train/weight
        '''
        Return range bounds of total observation/action space

        :param primitives: (dict) obs/act/structure info of primitives
        :return: ([2x2 np.array]): 
            dims[0][0] = obs min np.array, dims[0][1] = obs max np.array
            dims[1][0] = act min np.array, dims[1][1] = act max np.array
        '''
        weight_name = 'level'+str(level)+'_'+composite_primitive_name+"/weight"
        obs_dim = primitives[weight_name]['obs'][1][-1] + 1  # dimension = last index + 1
        obs_min_array = np.array([-float('inf')]*obs_dim)
        obs_max_array = np.array([float('inf')]*obs_dim)

        act_dim = 0
        for name, info_dict in primitives.items():
            if name != 'pretrained_param' and 'weight' not in name.split('/'):
                act_dim = max(act_dim, info_dict['act'][1][-1]+1)
        act_min_array = np.array([-float('inf')]*act_dim)
        act_max_array = np.array([float('inf')]*act_dim)

        for name, info_dict in primitives.items():
            if name != 'pretrained_param' and 'weight' not in name.split('/'):
                print("update prim: ", name)
                for i, idx in enumerate(info_dict['obs'][1]):
                    obs_min_prim = info_dict['obs'][0].low[i]
                    obs_max_prim = info_dict['obs'][0].high[i]
                    if obs_min_array[idx] not in [-float('inf'), obs_min_prim]:
                        print("\n\t\033[93m[WARNING]: You are about to overwrite dim{2} min bound of obs[{0:2.3f}] with {1:2.3f}\033[0m".format(obs_min_array[idx], obs_min_prim, idx))
                    obs_min_array[idx] = obs_min_prim
                    if obs_max_array[idx] not in [float('inf'), obs_max_prim]:
                        print("\n\t\033[93m[WARNING]: You are about to overwrite dim{2} max bound of obs[{0:2.3f}] with {1:2.3f}\033[0m".format(obs_max_array[idx], obs_max_prim, idx))
                    obs_max_array[idx] = obs_max_prim
                for i, idx in enumerate(info_dict['act'][1]):
                    act_min_prim = info_dict['act'][0].low[i]
                    act_max_prim = info_dict['act'][0].high[i]
                    if act_min_array[idx] not in [-float('inf'), act_min_prim]:
                        print("\n\t\033[93m[WARNING]: You are about to overwrite dim{2} min bound of act[{0:2.3f}] with {1:2.3f}\033[0m".format(act_min_array[idx], act_min_prim, idx))
                    act_min_array[idx] = act_min_prim
                    if act_max_array[idx] not in [float('inf'), act_max_prim]:
                        print("\n\t\033[93m[WARNING]: You are about to overwrite dim{2} max bound of act[{0:2.3f}] with {1:2.3f}\033[0m".format(act_max_array[idx], act_max_prim, idx))
                    act_max_array[idx] = act_max_prim
            ranges = [[obs_min_array, obs_max_array], [act_min_array, act_max_array]]

        return ranges

    @staticmethod
    def _save_to_file_cloudpickle(save_path, data=None, params=None):
        """Legacy code for saving models with cloudpickle

        :param save_path: (str or file-like) Where to store the model
        :param data: (OrderedDict) Class parameters being stored
        :param params: (OrderedDict) Model parameters being stored
        """
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".pkl"

            with open(save_path, "wb") as file_:
                cloudpickle.dump((data, params), file_)
        else:
            # Here save_path is a file-like object, not a path
            cloudpickle.dump((data, params), save_path)

    @staticmethod
    def _save_to_file_zip(save_path, data=None, params=None):
        """Save model to a .zip archive

        :param save_path: (str or file-like) Where to store the model
        :param data: (OrderedDict) Class parameters being stored
        :param params: (OrderedDict) Model parameters being stored
        """
        # data/params can be None, so do not
        # try to serialize them blindly
        if data is not None:
            serialized_data = data_to_json(data)
        if params is not None:
            serialized_params = params_to_bytes(params)
            # We also have to store list of the parameters
            # to store the ordering for OrderedDict.
            # We can trust these to be strings as they
            # are taken from the Tensorflow graph.
            serialized_param_list = json.dumps(
                list(params.keys()),
                indent=4
            )

        # Check postfix if save_path is a string
        if isinstance(save_path, str):
            _, ext = os.path.splitext(save_path)
            if ext == "":
                save_path += ".zip"

        # Create a zip-archive and write our objects
        # there. This works when save_path
        # is either str or a file-like
        with zipfile.ZipFile(save_path, "w") as file_:
            # Do not try to save "None" elements
            if data is not None:
                file_.writestr("data", serialized_data)
            if params is not None:
                file_.writestr("parameters", serialized_params)
                file_.writestr("parameter_list", serialized_param_list)

    @staticmethod
    def _save_to_file(save_path, data=None, params=None, cloudpickle=False):
        """Save model to a zip archive or cloudpickle file.

        :param save_path: (str or file-like) Where to store the model
        :param data: (OrderedDict) Class parameters being stored
        :param params: (OrderedDict) Model parameters being stored
        :param cloudpickle: (bool) Use old cloudpickle format
            (stable-baselines<=2.7.0) instead of a zip archive.
        """
        if cloudpickle:
            BaseRLModel._save_to_file_cloudpickle(save_path, data, params)
        else:
            BaseRLModel._save_to_file_zip(save_path, data, params)

    @staticmethod
    def _load_from_file_cloudpickle(load_path):
        """Legacy code for loading older models stored with cloudpickle

        :param load_path: (str or file-like) where from to load the file
        :return: (dict, OrderedDict) Class parameters and model parameters
        """
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".pkl"):
                    load_path += ".pkl"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

            with open(load_path, "rb") as file_:
                data, params = cloudpickle.load(file_)
        else:
            # Here load_path is a file-like object, not a path
            data, params = cloudpickle.load(load_path)

        return data, params

    @staticmethod
    def _load_from_file(load_path, load_data=True, custom_objects=None):
        """Load model data from a .zip archive

        :param load_path: (str or file-like) Where to load model from
        :param load_data: (bool) Whether we should load and return data
            (class parameters). Mainly used by `load_parameters` to
            only load model parameters (weights).
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :return: (dict, OrderedDict) Class parameters and model parameters
        """
        # Check if file exists if load_path is
        # a string
        if isinstance(load_path, str):
            if not os.path.exists(load_path):
                if os.path.exists(load_path + ".zip"):
                    load_path += ".zip"
                else:
                    raise ValueError("Error: the file {} could not be found".format(load_path))

        # Open the zip archive and load data.
        try:
            with zipfile.ZipFile(load_path, "r") as file_:
                namelist = file_.namelist()
                # If data or parameters is not in the
                # zip archive, assume they were stored
                # as None (_save_to_file allows this).
                data = None
                params = None
                if "data" in namelist and load_data:
                    # Load class parameters and convert to string
                    # (Required for json library in Python 3.5)
                    json_data = file_.read("data").decode()
                    data = json_to_data(json_data, custom_objects=custom_objects)

                if "parameters" in namelist:
                    # Load parameter list and and parameters
                    parameter_list_json = file_.read("parameter_list").decode()
                    parameter_list = json.loads(parameter_list_json)
                    serialized_params = file_.read("parameters")
                    params = bytes_to_params(
                        serialized_params, parameter_list
                    )
        except zipfile.BadZipFile:
            # load_path wasn't a zip file. Possibly a cloudpickle
            # file. Show a warning and fall back to loading cloudpickle.
            warnings.warn("It appears you are loading from a file with old format. " +
                          "Older cloudpickle format has been replaced with zip-archived " +
                          "models. Consider saving the model with new format.",
                          DeprecationWarning)
            # Attempt loading with the cloudpickle format.
            # If load_path is file-like, seek back to beginning of file
            if not isinstance(load_path, str):
                load_path.seek(0)
            data, params = BaseRLModel._load_from_file_cloudpickle(load_path)

        return data, params

    @staticmethod
    def _softmax(x_input):
        """
        An implementation of softmax.

        :param x_input: (numpy float) input vector
        :return: (numpy float) output vector
        """
        x_exp = np.exp(x_input.T - np.max(x_input.T, axis=0))
        return (x_exp / x_exp.sum(axis=0)).T

    @staticmethod
    def _is_vectorized_observation(observation, observation_space):
        """
        For every observation type, detects and validates the shape,
        then returns whether or not the observation is vectorized.

        :param observation: (np.ndarray) the input observation to validate
        :param observation_space: (gym.spaces) the observation space
        :return: (bool) whether the given observation is vectorized or not
        """
        if isinstance(observation_space, gym.spaces.Box):
            if observation.shape == observation_space.shape:
                return False
            elif observation.shape[1:] == observation_space.shape:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for ".format(observation.shape) +
                                 "Box environment, please use {} ".format(observation_space.shape) +
                                 "or (n_env, {}) for the observation shape."
                                 .format(", ".join(map(str, observation_space.shape))))
        elif isinstance(observation_space, gym.spaces.Discrete):
            if observation.shape == ():  # A numpy array of a number, has shape empty tuple '()'
                return False
            elif len(observation.shape) == 1:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for ".format(observation.shape) +
                                 "Discrete environment, please use (1,) or (n_env, 1) for the observation shape.")
        elif isinstance(observation_space, gym.spaces.MultiDiscrete):
            if observation.shape == (len(observation_space.nvec),):
                return False
            elif len(observation.shape) == 2 and observation.shape[1] == len(observation_space.nvec):
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for MultiDiscrete ".format(observation.shape) +
                                 "environment, please use ({},) or ".format(len(observation_space.nvec)) +
                                 "(n_env, {}) for the observation shape.".format(len(observation_space.nvec)))
        elif isinstance(observation_space, gym.spaces.MultiBinary):
            if observation.shape == (observation_space.n,):
                return False
            elif len(observation.shape) == 2 and observation.shape[1] == observation_space.n:
                return True
            else:
                raise ValueError("Error: Unexpected observation shape {} for MultiBinary ".format(observation.shape) +
                                 "environment, please use ({},) or ".format(observation_space.n) +
                                 "(n_env, {}) for the observation shape.".format(observation_space.n))
        else:
            raise ValueError("Error: Cannot determine if the observation is vectorized with the space type {}."
                             .format(observation_space))


class ActorCriticRLModel(BaseRLModel):
    """
    The base class for Actor critic model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param policy_base: (BasePolicy) the base policy used by this method (default=ActorCriticPolicy)
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, _init_setup_model, verbose=0, policy_base=ActorCriticPolicy,
                 requires_vec_env=False, policy_kwargs=None, seed=None, n_cpu_tf_sess=None):
        super(ActorCriticRLModel, self).__init__(policy, env, verbose=verbose, requires_vec_env=requires_vec_env,
                                                 policy_base=policy_base, policy_kwargs=policy_kwargs,
                                                 seed=seed, n_cpu_tf_sess=n_cpu_tf_sess)

        self.sess = None
        self.initial_state = None
        self.step = None
        self.proba_step = None
        self.params = None
        self._runner = None

    def _make_runner(self) -> AbstractEnvRunner:
        """Builds a new Runner.

        Lazily called whenever `self.runner` is accessed and `self._runner is None`.
        """
        raise NotImplementedError("This model is not configured to use a Runner")

    @property
    def runner(self) -> AbstractEnvRunner:
        if self._runner is None:
            self._runner = self._make_runner()
        return self._runner

    def set_env(self, env):
        self._runner = None  # New environment invalidates `self._runner`.
        super().set_env(env)

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def learn(self, total_timesteps, callback=None,
              log_interval=100, tb_log_name="run", reset_num_timesteps=True):
        pass

    def predict(self, observation, state=None, mask=None, deterministic=False):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions, _, states, _ = self.step(observation, state, mask, deterministic=deterministic)
        
        clipped_actions = actions
        # Clip the actions to avoid out of bound error
        # if self.act_model.squash:
        #     if isinstance(self.env.action_space, gym.spaces.Box):
        #         clipped_actions = unscale_action(self.env.action_space, clipped_actions)
        # else:
        #     if self.act_model.box_dist=='beta':
        #         if isinstance(self.env.action_space, gym.spaces.Box):
        #             if np.any(np.logical_or(clipped_actions > 1, clipped_actions<0)):
        #                 print("WARNING: clipped action have an invalid value of",clipped_actions)
        #             clipped_actions = unscale_action(self.env.action_space, clipped_actions*2-1)
        #     else:
        #         if isinstance(self.env.action_space, gym.spaces.Box):
        #             clipped_actions = np.clip(actions, self.env.action_space.low, self.env.action_space.high)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            clipped_actions = clipped_actions[0]

        return clipped_actions, states


    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if state is None:
            state = self.initial_state
        if mask is None:
            mask = [False for _ in range(self.n_envs)]
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions_proba = self.proba_step(observation, state, mask)

        if len(actions_proba) == 0:  # empty list means not implemented
            warnings.warn("Warning: action probability is not implemented for {} action space. Returning None."
                          .format(type(self.action_space).__name__))
            return None

        if actions is not None:  # comparing the action distribution, to given actions
            prob = None
            logprob = None
            actions = np.array([actions])
            if isinstance(self.action_space, gym.spaces.Discrete):
                actions = actions.reshape((-1,))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                prob = actions_proba[np.arange(actions.shape[0]), actions]

            elif isinstance(self.action_space, gym.spaces.MultiDiscrete):
                actions = actions.reshape((-1, len(self.action_space.nvec)))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                # Discrete action probability, over multiple categories
                actions = np.swapaxes(actions, 0, 1)  # swap axis for easier categorical split
                prob = np.prod([proba[np.arange(act.shape[0]), act]
                                         for proba, act in zip(actions_proba, actions)], axis=0)

            elif isinstance(self.action_space, gym.spaces.MultiBinary):
                actions = actions.reshape((-1, self.action_space.n))
                assert observation.shape[0] == actions.shape[0], \
                    "Error: batch sizes differ for actions and observations."
                # Bernoulli action probability, for every action
                prob = np.prod(actions_proba * actions + (1 - actions_proba) * (1 - actions), axis=1)

            elif isinstance(self.action_space, gym.spaces.Box):
                actions = actions.reshape((-1, ) + self.action_space.shape)
                mean, logstd = actions_proba
                std = np.exp(logstd)

                n_elts = np.prod(mean.shape[1:])  # first dimension is batch size
                log_normalizer = n_elts/2 * np.log(2 * np.pi) + 1/2 * np.sum(logstd, axis=1)

                # Diagonal Gaussian action probability, for every action
                logprob = -np.sum(np.square(actions - mean) / (2 * std), axis=1) - log_normalizer

            else:
                warnings.warn("Warning: action_probability not implemented for {} actions space. Returning None."
                              .format(type(self.action_space).__name__))
                return None

            # Return in space (log or normal) requested by user, converting if necessary
            if logp:
                if logprob is None:
                    logprob = np.log(prob)
                ret = logprob
            else:
                if prob is None:
                    prob = np.exp(logprob)
                ret = prob

            # normalize action proba shape for the different gym spaces
            ret = ret.reshape((-1, 1))
        else:
            ret = actions_proba

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            ret = ret[0]

        return ret

    def get_parameter_list(self):
        return self.params

    @abstractmethod
    def save(self, save_path, cloudpickle=False):
        pass

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Environment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)

        model.set_env(env)
        model.setup_model()

        model.load_parameters(params)

        return model


class OffPolicyRLModel(BaseRLModel):
    """
    The base class for off policy RL model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param replay_buffer: (ReplayBuffer) the type of replay buffer
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    :param policy_base: (BasePolicy) the base policy used by this method
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, replay_buffer=None, _init_setup_model=False, verbose=0, tensorboard_log=None,
                 requires_vec_env=False, policy_base=None,
                 policy_kwargs=None, seed=None, n_cpu_tf_sess=None, composite_primitive_name=None):
        super(OffPolicyRLModel, self).__init__(policy, env, verbose=verbose, requires_vec_env=requires_vec_env,
                                               policy_base=policy_base, policy_kwargs=policy_kwargs,
                                               seed=seed, n_cpu_tf_sess=n_cpu_tf_sess, composite_primitive_name=composite_primitive_name)

        self.replay_buffer = replay_buffer
        self.tensorboard_log = None

    @abstractmethod
    def setup_model(self):
        pass

    @abstractmethod
    def learn(self, total_timesteps, callback=None,
              log_interval=100, tb_log_name="run", reset_num_timesteps=True, replay_wrapper=None):
        pass

    @abstractmethod
    def predict(self, observation, state=None, mask=None, deterministic=False):
        pass

    @abstractmethod
    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        pass

    @abstractmethod
    def save(self, save_path, cloudpickle=False):
        pass

    @classmethod
    def load(cls, load_path, env=None, custom_objects=None, **kwargs):
        """
        Load the model from file

        :param load_path: (str or file-like) the saved parameter location
        :param env: (Gym Environment) the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model)
        :param custom_objects: (dict) Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            `keras.models.load_model`. Useful when you have an object in
            file that can not be deserialized.
        :param kwargs: extra arguments to change the model when loading
        """
        data, params = cls._load_from_file(load_path, custom_objects=custom_objects)

        if 'policy_kwargs' in kwargs and kwargs['policy_kwargs'] != data['policy_kwargs']:
            raise ValueError("The specified policy kwargs do not equal the stored policy kwargs. "
                             "Stored kwargs: {}, specified kwargs: {}".format(data['policy_kwargs'],
                                                                              kwargs['policy_kwargs']))

        model = cls(policy=data["policy"], env=None, _init_setup_model=False)
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        
        # NOTE: Once loaded, type of policy is fixed
        model.set_env(env)
        model.setup_model()

        model.load_parameters(params, exact_match=False)

        return model


class _UnvecWrapper(VecEnvWrapper):
    def __init__(self, venv):
        """
        Unvectorize a vectorized environment, for vectorized environment that only have one environment

        :param venv: (VecEnv) the vectorized environment to wrap
        """
        super().__init__(venv)
        assert venv.num_envs == 1, "Error: cannot unwrap a environment wrapper that has more than one environment."

    def seed(self, seed=None):
        return self.venv.env_method('seed', seed)

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.venv, attr)

    def __set_attr__(self, attr, value):
        if attr in self.__dict__:
            setattr(self, attr, value)
        else:
            setattr(self.venv, attr, value)

    def compute_reward(self, achieved_goal, desired_goal, _info):
        return float(self.venv.env_method('compute_reward', achieved_goal, desired_goal, _info)[0])

    @staticmethod
    def unvec_obs(obs):
        """
        :param obs: (Union[np.ndarray, dict])
        :return: (Union[np.ndarray, dict])
        """
        if not isinstance(obs, dict):
            return obs[0]
        obs_ = OrderedDict()
        for key in obs.keys():
            obs_[key] = obs[key][0]
        del obs
        return obs_

    def reset(self):
        return self.unvec_obs(self.venv.reset())

    def step_async(self, actions):
        self.venv.step_async([actions])

    def step_wait(self):
        obs, rewards, dones, information = self.venv.step_wait()
        return self.unvec_obs(obs), float(rewards[0]), dones[0], information[0]

    def render(self, mode='human'):
        return self.venv.render(mode=mode)


class SetVerbosity:
    def __init__(self, verbose=0):
        """
        define a region of code for certain level of verbosity

        :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        """
        self.verbose = verbose

    def __enter__(self):
        self.tf_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
        self.log_level = logger.get_level()
        self.gym_level = gym.logger.MIN_LEVEL

        if self.verbose <= 1:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        if self.verbose <= 0:
            logger.set_level(logger.DISABLED)
            gym.logger.set_level(gym.logger.DISABLED)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.verbose <= 1:
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.tf_level

        if self.verbose <= 0:
            logger.set_level(self.log_level)
            gym.logger.set_level(self.gym_level)


class TensorboardWriter:
    def __init__(self, graph, tensorboard_log_path, tb_log_name, new_tb_log=True):
        """
        Create a Tensorboard writer for a code segment, and saves it to the log directory as its own run

        :param graph: (Tensorflow Graph) the model graph
        :param tensorboard_log_path: (str) the save path for the log (can be None for no logging)
        :param tb_log_name: (str) the name of the run for tensorboard log
        :param new_tb_log: (bool) whether or not to create a new logging folder for tensorbaord
        """
        self.graph = graph
        self.tensorboard_log_path = tensorboard_log_path
        self.tb_log_name = tb_log_name
        self.writer = None
        self.new_tb_log = new_tb_log

    def __enter__(self):
        if self.tensorboard_log_path is not None:
            latest_run_id = self._get_latest_run_id()
            if self.new_tb_log:
                latest_run_id = latest_run_id + 1
            save_path = os.path.join(self.tensorboard_log_path, "{}_{}".format(self.tb_log_name, latest_run_id))
            self.writer = tf.summary.FileWriter(save_path, graph=self.graph)
        return self.writer

    def _get_latest_run_id(self):
        """
        returns the latest run number for the given log name and log path,
        by finding the greatest number in the directories.

        :return: (int) latest run number
        """
        max_run_id = 0
        for path in glob.glob("{}/{}_[0-9]*".format(self.tensorboard_log_path, self.tb_log_name)):
            file_name = path.split(os.sep)[-1]
            ext = file_name.split("_")[-1]
            if self.tb_log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
                max_run_id = int(ext)
        return max_run_id

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.writer is not None:
            self.writer.add_graph(self.graph)
            self.writer.flush()
