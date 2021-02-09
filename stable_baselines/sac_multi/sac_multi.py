import sys
import time
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python import debug as tf_debug

from stable_baselines.common import tf_util, OffPolicyRLModel, SetVerbosity, TensorboardWriter, zipsame
from stable_baselines.common.radam import RAdamOptimizer
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.cg import conjugate_gradient
from stable_baselines.common.math_util import safe_mean, unscale_action, scale_action
from stable_baselines.common.schedules import get_schedule_fn
from stable_baselines.common.buffers import ReplayBuffer
from stable_baselines.sac_multi.policies import SACPolicy
from stable_baselines import logger


class SAC_MULTI(OffPolicyRLModel):
    """
    Soft Actor-Critic (SAC)
    Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor,
    This implementation borrows code from original implementation (https://github.com/haarnoja/sac)
    from OpenAI Spinning Up (https://github.com/openai/spinningup) and from the Softlearning repo
    (https://github.com/rail-berkeley/softlearning/)
    Paper: https://arxiv.org/abs/1801.01290
    Introduction to SAC: https://spinningup.openai.com/en/latest/algorithms/sac.html

    :param policy: (SACPolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) the discount factor
    :param learning_rate: (float or callable) learning rate for adam optimizer,
        the same learning rate will be used for all networks (Q-Values, Actor and Value function)
        it can be a function of the current progress (from 1 to 0)
    :param buffer_size: (int) size of the replay buffer
    :param batch_size: (int) Minibatch size for each gradient update
    :param tau: (float) the soft update coefficient ("polyak update", between 0 and 1)
    :param ent_coef: (str or float) Entropy regularization coefficient. (Equivalent to
        inverse of reward scale in the original SAC paper.)  Controlling exploration/exploitation trade-off.
        Set it to 'auto' to learn it automatically (and 'auto_0.1' for using 0.1 as initial value)
    :param train_freq: (int) Update the model every `train_freq` steps.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_update_interval: (int) update the target network every `target_network_update_freq` steps.
    :param gradient_steps: (int) How many gradient update after each step
    :param target_entropy: (str or float) target entropy when learning ent_coef (ent_coef = 'auto')
    :param action_noise: (ActionNoise) the action noise type (None by default), this can help
        for hard exploration problem. Cf DDPG for the different action noise type.
    :param random_exploration: (float) Probability of taking a random action (as in an epsilon-greedy strategy)
        This is not needed for SAC normally but can help exploring when using HER + SAC.
        This hack was present in the original OpenAI Baselines repo (DDPG + HER)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    :param policy_kwargs: (dict) additional arguments to be passed to the policy on creation
    :param full_tensorboard_log: (bool) enable additional logging when using tensorboard
        Note: this has no effect on SAC logging for now
    :param seed: (int) Seed for the pseudo-random generators (python, numpy, tensorflow).
        If None (default), use random seed. Note that if you want completely deterministic
        results, you must set `n_cpu_tf_sess` to 1.
    :param n_cpu_tf_sess: (int) The number of threads for TensorFlow operations
        If None, the number of cpu of the current machine will be used.
    """

    def __init__(self, policy, env, layers={}, gamma=0.99, learning_rate=1e-4, buffer_size=50000,
                 replay_buffer=None, learning_starts=5000, train_freq=1, batch_size=64,
                 tau=0.005, ent_coef='auto', target_update_interval=1,
                 gradient_steps=1, target_entropy='auto', box_dist='gaussian', action_noise=None,
                 random_exploration=0.0, verbose=0, tensorboard_log=None,
                 _init_setup_model=True, policy_kwargs=None, full_tensorboard_log=False,
                 seed=None, n_cpu_tf_sess=None, composite_primitive_name=None):

        super(SAC_MULTI, self).__init__(policy=policy, env=env, replay_buffer=replay_buffer, verbose=verbose,
                                  policy_base=SACPolicy, requires_vec_env=False, policy_kwargs=policy_kwargs,
                                  seed=seed, n_cpu_tf_sess=n_cpu_tf_sess, composite_primitive_name=composite_primitive_name)

        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.tau = tau
        # In the original paper, same learning rate is used for all networks
        # self.policy_lr = learning_rate
        # self.qf_lr = learning_rate
        # self.vf_lr = learning_rate
        # Entropy coefficient / Entropy temperature
        # Inverse of the reward scale
        self.ent_coef = ent_coef
        self.target_update_interval = target_update_interval
        self.gradient_steps = gradient_steps
        self.gamma = gamma
        self.action_noise = action_noise
        self.random_exploration = random_exploration

        self.value_fn = None
        self.graph = None
        self.replay_buffer = replay_buffer
        self.sess = None
        self.tensorboard_log = tensorboard_log
        self.verbose = verbose
        self.params = None
        self.summary = None
        self.policy_tf = None
        self.box_dist = box_dist
        self.layers = layers
        self.target_entropy = target_entropy
        self.full_tensorboard_log = full_tensorboard_log

        self.obs_target = None
        self.target_policy = None
        self.actions_ph = None
        self.rewards_ph = None
        self.terminals_ph = None
        self.observations_ph = None
        self.action_target = None
        self.next_observations_ph = None
        self.value_target = None
        self.step_ops = None
        self.target_update_op = None
        self.infos_names = None
        self.entropy = None
        self.target_params = None
        self.learning_rate_ph = None
        self.processed_obs_ph = None
        self.processed_next_obs_ph = None
        self.log_ent_coef = None
        self.grad_logger = True

        if _init_setup_model:
            self.setup_model()

    def _get_pretrain_placeholders(self):
        policy = self.policy_tf
        print("in pretrain placeholders")
        # Rescale
        deterministic_action = unscale_action(self.action_space, self.deterministic_action)
        return policy.obs_ph, self.actions_ph, deterministic_action

    def setup_model(self):
        print("Setting up model")
        print(self.policy_kwargs)
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)
                if self.replay_buffer is None:
                    self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    # Create policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space, layers=self.layers,
                                                **self.policy_kwargs)
                    old_policy_tf = self.policy(self.sess, self.observation_space, self.action_space, layers=self.layers,
                                                **self.policy_kwargs)
                    comp_policy_tf = self.policy(self.sess, self.observation_space, self.action_space, layers=self.layers,
                                                **self.policy_kwargs)

                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space, layers=self.layers,
                                                     **self.policy_kwargs)

                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    self.next_observations_ph = self.target_policy.obs_ph
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    self.action_target = self.target_policy.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                     name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                with tf.variable_scope("model", reuse=False):
                    # Create the policy
                    # first return value corresponds to deterministic actions
                    # policy_out corresponds to stochastic actions, used for training
                    # logp_pi is the log probability of actions taken by the policy
                    self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_actor(self.processed_obs_ph)
                    # Monitor the entropy of the policy,
                    # this is not used for training
                    self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                    #  Use two Q-functions to improve performance by reducing overestimation bias.
                    qf1, qf2, value_fn = self.policy_tf.make_critics(self.processed_obs_ph, self.actions_ph,
                                                                     create_qf=True, create_vf=True)
                    qf1_pi, qf2_pi, _ = self.policy_tf.make_critics(self.processed_obs_ph,
                                                                    policy_out, create_qf=True, create_vf=False,
                                                                    reuse=True)

                    # Target entropy is used when learning the entropy coefficient
                    if self.target_entropy == 'auto':
                        # automatically set target entropy if needed
                        self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                    else:
                        # Force conversion
                        # this will also throw an error for unexpected string
                        self.target_entropy = float(self.target_entropy)

                    # The entropy coefficient or entropy can be learned automatically
                    # see Automating Entropy Adjustment for Maximum Entropy RL section
                    # of https://arxiv.org/abs/1812.05905
                    if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                        # Default initial value of ent_coef when learned
                        init_value = 1.0
                        if '_' in self.ent_coef:
                            init_value = float(self.ent_coef.split('_')[1])
                            assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                        self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                            initializer=np.log(init_value).astype(np.float32))
                        self.ent_coef = tf.exp(self.log_ent_coef)
                    else:
                        # Force conversion to float
                        # this will throw an error if a malformed string (different from 'auto')
                        # is passed
                        self.ent_coef = float(self.ent_coef)

                with tf.variable_scope("target", reuse=False):
                    # Create the value network
                    _, _, value_target = self.target_policy.make_critics(self.processed_next_obs_ph,
                                                                         create_qf=False, create_vf=True)
                    self.value_target = value_target

                with tf.variable_scope("loss", reuse=False):
                    # Take the min of the two Q-Values (Double-Q Learning)
                    min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                    # Target for Q value regression
                    q_backup = tf.stop_gradient(
                        self.rewards_ph +
                        (1 - self.terminals_ph) * self.gamma * self.value_target
                    )

                    # Compute Q-Function loss
                    # TODO: test with huber loss (it would avoid too high values)
                    qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                    qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                    # Compute the entropy temperature loss
                    # it is used when the entropy coefficient is learned
                    ent_coef_loss, entropy_optimizer = None, None
                    if not isinstance(self.ent_coef, float):
                        ent_coef_loss = -tf.reduce_mean(
                            self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                        entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)

                    # Compute the policy loss
                    # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                    policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)

                    # NOTE: in the original implementation, they have an additional
                    # regularization loss for the Gaussian parameters
                    # this is not used for now
                    # policy_loss = (policy_kl_loss + policy_regularization_loss)
                    policy_loss = policy_kl_loss

                    # Target for value fn regression
                    # We update the vf towards the min of two Q-functions in order to
                    # reduce overestimation bias from function approximation error.
                    v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                    value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)
                    values_losses = qf1_loss + qf2_loss + value_loss

                    # Policy train op
                    # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                    policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph, epsilon=1e-6)
                    #policy_optimizer = RAdamOptimizer(learning_rate=self.learning_rate_ph, beta1=0.9, beta2=0.999, weight_decay=0.0)

                    policy_var_new = tf_util.get_trainable_vars('model/pi')
                    policy_var_old = tf_util.get_trainable_vars('model/old_policy/pi')
                    policy_var_comp = tf_util.get_trainable_vars('model/comp_policy/pi')
                    #policy_var_list = policy_var_new + policy_var_old
                    policy_var_list = policy_var_new

                    if self.grad_logger:
                        self.grad_logger_op = {}
                        for index, grad in enumerate(grads):
                            print(grad)
                            self.grad_logger_op[grad[0].name] = grad[0]
                            self.grad_logger_op[grad[1].name] = grad[1]

                    self.policy_train_op = policy_train_op = policy_optimizer.apply_gradients(grads)

                    # Value train op
                    value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                    values_params = tf_util.get_trainable_vars('model/values_fn')

                    source_params = tf_util.get_trainable_vars("model/values_fn/vf")
                    target_params = tf_util.get_trainable_vars("target/values_fn/vf")

                    # Polyak averaging for target variables
                    self.target_update_op = [
                        tf.assign(target, (1 - self.tau) * target + self.tau * source)
                        for target, source in zip(target_params, source_params)
                    ]
                    # Initializing target to match source variables
                    target_init_op = [
                        tf.assign(target, source)
                        for target, source in zip(target_params, source_params)
                    ]

                    self.assign_old_new_op = [
                        tf.assign(var_old, var_new)
                        for var_old, var_new in zip(policy_var_old, policy_var_new)
                    ]

                    self.assign_comp_new_op = [
                        tf.assign(var_comp, var_new)
                        for var_comp, var_new in zip(policy_var_comp, policy_var_new)
                    ]

                    # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                    # and we first need to compute the policy action before computing q values losses
                    with tf.control_dependencies([policy_train_op]):
                        train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                        self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                        # All ops to call during one training step
                        self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                         value_loss, qf1, qf2, value_fn, logp_pi,
                                         self.entropy, policy_train_op, train_values_op]

                        # Add entropy coefficient optimization operation if needed
                        if ent_coef_loss is not None:
                            with tf.control_dependencies([train_values_op]):
                                ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                self.infos_names += ['ent_coef_loss', 'ent_coef']
                                self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

                    # Monitor losses and entropy in tensorboard
                    tf.summary.scalar('policy_loss', policy_loss)
                    tf.summary.scalar('qf1_loss', qf1_loss)
                    tf.summary.scalar('qf2_loss', qf2_loss)
                    tf.summary.scalar('value_loss', value_loss)
                    tf.summary.scalar('entropy', self.entropy)
                    if ent_coef_loss is not None:
                        tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                        tf.summary.scalar('ent_coef', self.ent_coef)

                    tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = tf_util.get_trainable_vars("model")
                self.pi_params = tf_util.get_trainable_vars("model/pi")
                self.target_params = tf_util.get_trainable_vars("target/values_fn/vf")
                variable_list = tf_util.get_trainable_vars('') + tf_util.get_globals_vars('')
                self.get_variable_op = {}
                for var in variable_list:
                    if 'values_fn' not in var.name:
                        self.get_variable_op[var.name] = var
                # print(self.get_variable_op)

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def setup_custom_model(self, primitives, load_value=True):
        with SetVerbosity(self.verbose):
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.set_random_seed(self.seed)
                self.sess = tf_util.make_session(num_cpu=self.n_cpu_tf_sess, graph=self.graph)
                self.replay_buffer = ReplayBuffer(self.buffer_size)

                with tf.variable_scope("input", reuse=False):
                    # Create custom policy and target TF objects
                    self.policy_tf = self.policy(self.sess, self.observation_space, self.action_space, layers=self.layers,
                                                **self.policy_kwargs)
                    self.target_policy = self.policy(self.sess, self.observation_space, self.action_space, layers=self.layers,
                                                    **self.policy_kwargs)
                    
                    assert not self.policy_tf.layer_norm, "\033[91m[ERROR]: LayerNormalized policy is not supported for now. Try to train policy with unnormalized layers\033[0m"
                    
                    # Initialize Placeholders
                    self.observations_ph = self.policy_tf.obs_ph
                    # Normalized observation for pixels
                    self.processed_obs_ph = self.policy_tf.processed_obs
                    #tf.summary.histogram('observations', self.processed_obs_ph)
                    self.next_observations_ph = self.target_policy.obs_ph
                    self.processed_next_obs_ph = self.target_policy.processed_obs
                    # None
                    self.action_target = self.target_policy.action_ph
                    self.terminals_ph = tf.placeholder(tf.float32, shape=(None, 1), name='terminals')
                    self.rewards_ph = tf.placeholder(tf.float32, shape=(None, 1), name='rewards')
                    self.actions_ph = tf.placeholder(tf.float32, shape=(None,) + self.action_space.shape,
                                                    name='actions')
                    self.learning_rate_ph = tf.placeholder(tf.float32, [], name="learning_rate_ph")

                loaded = True if 'loaded' in primitives.keys() else False
                if loaded:
                    with tf.variable_scope("model", reuse=False):
                        self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_custom_actor(self.processed_obs_ph, primitives, self.tails, self.action_space.shape[0], scope='pi/loaded')
                else:
                    with tf.variable_scope("model", reuse=False):
                        # Create the policy
                        # first return value corresponds to deterministic actions
                        # policy_out corresponds to stochastic actions, used for training
                        # logp_pi is the log probability of actions taken by the policy
                        self.deterministic_action, policy_out, logp_pi = self.policy_tf.make_custom_actor(self.processed_obs_ph, primitives, self.tails, self.action_space.shape[0])
            
                        # Monitor the entropy of the policy,
                        # this is not used for training
                        self.entropy = tf.reduce_mean(self.policy_tf.entropy)
                        
                        #  Use two Q-functions to improve performance by reducing overestimation bias.
                        qf1, qf2, value_fn = self.policy_tf.make_custom_critics(self.processed_obs_ph, self.actions_ph, primitives, self.tails,
                                                                        create_qf=True, create_vf=True)
                        qf1_pi, qf2_pi, _ = self.policy_tf.make_custom_critics(self.processed_obs_ph, policy_out, primitives, self.tails,
                                                                        create_qf=True, create_vf=False,
                                                                        reuse=True)

                        # Target entropy is used when learning the entropy coefficient
                        if self.target_entropy == 'auto':
                            # automatically set target entropy if needed
                            self.target_entropy = -np.prod(self.action_space.shape).astype(np.float32)
                        else:
                            # Force conversion
                            # this will also throw an error for unexpected string
                            self.target_entropy = float(self.target_entropy)

                        # The entropy coefficient or entropy can be learned automatically
                        # see Automating Entropy Adjustment for Maximum Entropy RL section
                        # of https://arxiv.org/abs/1812.05905
                        if isinstance(self.ent_coef, str) and self.ent_coef.startswith('auto'):
                            # Default initial value of ent_coef when learned
                            init_value = 1.0
                            if '_' in self.ent_coef:
                                init_value = float(self.ent_coef.split('_')[1])
                                assert init_value > 0., "The initial value of ent_coef must be greater than 0"

                            self.log_ent_coef = tf.get_variable('log_ent_coef', dtype=tf.float32,
                                                                initializer=np.log(init_value).astype(np.float32))
                            self.ent_coef = tf.exp(self.log_ent_coef)
                        else:
                            # Force conversion to float
                            # this will throw an error if a malformed string (different from 'auto')
                            # is passed
                            self.ent_coef = float(self.ent_coef)

                    with tf.variable_scope("target", reuse=False):
                        # Create the value network
                        _, _, value_target = self.target_policy.make_custom_critics(self.processed_next_obs_ph, None, primitives, self.tails,
                                                                            create_qf=False, create_vf=True)
                        self.value_target = value_target

                    with tf.variable_scope("loss", reuse=False):
                        # Take the min of the two Q-Values (Double-Q Learning)
                        min_qf_pi = tf.minimum(qf1_pi, qf2_pi)

                        # Target for Q value regression
                        q_backup = tf.stop_gradient(
                            self.rewards_ph +
                            (1 - self.terminals_ph) * self.gamma * self.value_target
                        )

                        # Compute Q-Function loss
                        # TODO: test with huber loss (it would avoid too high values)
                        qf1_loss = 0.5 * tf.reduce_mean((q_backup - qf1) ** 2)
                        qf2_loss = 0.5 * tf.reduce_mean((q_backup - qf2) ** 2)

                        # Compute the entropy temperature loss
                        # it is used when the entropy coefficient is learned
                        ent_coef_loss, entropy_optimizer = None, None
                        if not isinstance(self.ent_coef, float):
                            ent_coef_loss = -tf.reduce_mean(
                                self.log_ent_coef * tf.stop_gradient(logp_pi + self.target_entropy))
                            entropy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                        # Compute the policy loss
                        # Alternative: policy_kl_loss = tf.reduce_mean(logp_pi - min_qf_pi)
                        qf1_pi = tf.reshape(qf1_pi,[-1])
                        logp_pi_mean = tf.reduce_mean(logp_pi)
                        qf1_pi_mean = tf.reduce_mean(qf1_pi)
                        
                        policy_kl_loss = tf.reduce_mean(self.ent_coef * logp_pi - qf1_pi)

                        # NOTE: in the original implementation, they have an additional
                        # regularization loss for the Gaussian parameters
                        # this is not used for now
                        # policy_loss = (policy_kl_loss + policy_regularization_loss)
                        policy_loss = policy_kl_loss

                        # Target for value fn regression
                        # We update the vf towards the min of two Q-functions in order to
                        # reduce overestimation bias from function approximation error.
                        v_backup = tf.stop_gradient(min_qf_pi - self.ent_coef * logp_pi)
                        value_loss = 0.5 * tf.reduce_mean((value_fn - v_backup) ** 2)

                        values_losses = qf1_loss + qf2_loss + value_loss

                        # Policy train op
                        # (has to be separate from value train op, because min_qf_pi appears in policy_loss)
                        policy_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                        # policy_optimizer = RAdamOptimizer(learning_rate=self.learning_rate_ph, beta1=0.9, beta2=0.999, weight_decay=0.0)
                        # NOTE: params of pretrained networks should not be fine-tuned to avoid forgetting
                        policy_var_list = tf_util.get_trainable_vars('model/pi/train')
                        # print("Policy optimizee: ")
                        # for var in policy_var_list:
                        #     tf.summary.histogram(var.name, var)
                        #     print("\t",var)
                        # policy_var_freeze = tf_util.get_trainable_vars('model/pi/freeze')
                        # print("Policy NOT optimizee: ")
                        # for var in policy_var_freeze:
                        #     tf.summary.histogram(var.name, var)
                        #     print("\t",var)

                        grads = policy_optimizer.compute_gradients(policy_loss, var_list=policy_var_list)
                        if self.grad_logger:
                            self.grad_logger_op = {}
                            for index, grad in enumerate(grads):
                                # self.grad_logger_op[grad[0].name] = grad[0]
                                self.grad_logger_op[grad[1].name] = grad[1]
                        policy_train_op = policy_optimizer.apply_gradients(grads)

                        # Value train op
                        value_optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate_ph)
                        values_params = tf_util.get_trainable_vars('model/values_fn/\w*/train')

                        source_params_trainable = tf_util.get_trainable_vars("model/values_fn/vf/train")
                        # print("Value source optimizee: ")
                        # for var in source_params_trainable:
                        #     print("\t",var)
                        target_params_trainable = tf_util.get_trainable_vars("target/values_fn/vf/train")
                        # print("Value target optimizee: ")
                        # for var in target_params_trainable:
                        #     print("\t",var)

                        source_params = tf_util.get_trainable_vars("model/values_fn/vf")
                        target_params = tf_util.get_trainable_vars("target/values_fn/vf")

                        # Polyak averaging for target variables
                        self.target_update_op = [
                            tf.assign(target, (1 - self.tau) * target + self.tau * source)
                            for target, source in zip(target_params_trainable, source_params_trainable)
                        ]
                        # Initializing target to match source variables
                        target_init_op = [
                            tf.assign(target, source)
                            for target, source in zip(target_params, source_params)
                        ]

                        # Control flow is used because sess.run otherwise evaluates in nondeterministic order
                        # and we first need to compute the policy action before computing q values losses
                        with tf.control_dependencies([policy_train_op]):
                            train_values_op = value_optimizer.minimize(values_losses, var_list=values_params)

                            self.infos_names = ['policy_loss', 'qf1_loss', 'qf2_loss', 'value_loss', 'entropy']
                            # All ops to call during one training step
                            self.step_ops = [policy_loss, qf1_loss, qf2_loss,
                                            value_loss, qf1, qf2, value_fn, logp_pi,
                                            self.entropy, policy_train_op, train_values_op]

                            # Add entropy coefficient optimization operation if needed
                            if ent_coef_loss is not None:
                                with tf.control_dependencies([train_values_op]):
                                    ent_coef_op = entropy_optimizer.minimize(ent_coef_loss, var_list=self.log_ent_coef)
                                    self.infos_names += ['ent_coef_loss', 'ent_coef']
                                    self.step_ops += [ent_coef_op, ent_coef_loss, self.ent_coef]

                        # Monitor losses and entropy in tensorboard
                        tf.summary.scalar('policy_loss', policy_loss)
                        tf.summary.scalar('qf1_loss', qf1_loss)
                        tf.summary.scalar('qf2_loss', qf2_loss)
                        tf.summary.scalar('value_loss', value_loss)
                        tf.summary.scalar('entropy', self.entropy)
                        # tf.summary.scalar('mean: qf1_pi', qf1_pi_mean)
                        tf.summary.scalar('mean: logp_pi', logp_pi_mean)

                        if ent_coef_loss is not None:
                            tf.summary.scalar('ent_coef_loss', ent_coef_loss)
                            tf.summary.scalar('ent_coef', self.ent_coef)
                            tf.summary.scalar('target_ent', self.target_entropy)
                        #tf.summary.scalar('learning_rate', tf.reduce_mean(self.learning_rate_ph))

                # Retrieve parameters that must be saved
                self.params = tf_util.get_trainable_vars("model")
                self.target_params = tf_util.get_trainable_vars("target/values_fn/vf")
                self.get_variable_op = tf_util.get_trainable_vars('') + tf_util.get_globals_vars('')

                # Initialize Variables and target network
                with self.sess.as_default():
                    self.sess.run(tf.global_variables_initializer())
                    if not loaded:
                        self.sess.run(target_init_op)

                self.summary = tf.summary.merge_all()

    def _train_step(self, step, writer, learning_rate):
        # Assign old policy to the new one
        # self.sess.run(self.assign_old_new_op)

        # Sample a batch from the replay buffer
        batch = self.replay_buffer.sample(self.batch_size, env=self._vec_normalize_env)
        batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones = batch

        feed_dict = {
            self.observations_ph: batch_obs,
            self.actions_ph: batch_actions,
            self.next_observations_ph: batch_next_obs,
            self.rewards_ph: batch_rewards.reshape(self.batch_size, -1),
            self.terminals_ph: batch_dones.reshape(self.batch_size, -1),
            self.learning_rate_ph: learning_rate
        }

        # Do one gradient step
        # and optionally compute log for tensorboard
        if writer is not None:
            if self.grad_logger:
                logger = self.sess.run(self.grad_logger_op, feed_dict)
                self.file_logger.writelines('######################### STEP: {}'.format(step)+' #########################\n')
                self.file_logger.writelines('DATA: '+str(np.mean(batch_obs))+', '+str(np.mean(batch_actions))+', '+str(np.mean(batch_rewards))+', '+str(np.mean(batch_next_obs))+'\n')
                self.file_mean_logger.writelines('######################### STEP: {}'.format(step)+' #########################\n')
                for name, item in logger.items():
                    self.file_logger.writelines(str(name)+'\n')
                    self.file_logger.writelines(str(np.mean(item, axis=len(item.shape[:])-1)))
                    self.file_logger.writelines('\n')
                    self.file_mean_logger.writelines(str(name)+'\n')
                    self.file_mean_logger.writelines(str(np.mean(item)))
                    self.file_mean_logger.writelines('\n')
                self.file_logger.writelines('\n')
                self.file_mean_logger.writelines('\n')

            out = self.sess.run([self.summary] + self.step_ops, feed_dict)
            summary = out.pop(0)
            writer.add_summary(summary, step)
        else:
            if self.grad_logger:
                logger = self.sess.run(self.grad_logger_op, feed_dict)
                variables = self.sess.run(self.get_variable_op, feed_dict)
                self.file_logger.writelines('######################### STEP: {}'.format(step)+' #########################\n')
                self.file_logger.writelines('DATA: '+str(np.mean(batch_obs))+', '+str(np.mean(batch_actions))+', '+str(np.mean(batch_rewards))+', '+str(np.mean(batch_next_obs))+'\n')
                self.file_mean_logger.writelines('######################### STEP: {}'.format(step)+' #########################\n')
                self.file_vars.writelines('######################### STEP: {}'.format(step)+' #########################\n')
                for name,item in logger.items():
                    self.file_logger.writelines(str(name)+'\n')
                    self.file_logger.writelines(str(np.mean(item, axis=len(item.shape[:])-1)))
                    self.file_logger.writelines('\n')
                    self.file_mean_logger.writelines(str(name)+'\n')
                    self.file_mean_logger.writelines(str(np.mean(item)))
                    self.file_mean_logger.writelines('\n')
                for name,item in variables.items():
                    self.file_vars.writelines(str(name)+'\n')
                    self.file_vars.writelines(str(np.mean(item)))
                    if np.any(np.isnan(item)):
                        self.file_vars.writelines('\nNAN in '+str(name)+'\n')
                        self.file_vars.writelines(str(item))
                    self.file_vars.writelines('\n')
                self.file_logger.writelines('\n')
                self.file_mean_logger.writelines('\n')
                self.file_vars.writelines('\n')
            out = self.sess.run(self.step_ops, feed_dict)

        # Unpack to monitor losses and entropy
        policy_loss, qf1_loss, qf2_loss, value_loss, *values = out
        # qf1, qf2, value_fn, logp_pi, entropy, *_ = values
        entropy = values[4]

        if self.log_ent_coef is not None:
            ent_coef_loss, ent_coef = values[-2:]
            return policy_loss, qf1_loss, qf2_loss, value_loss, entropy, ent_coef_loss, ent_coef

        return policy_loss, qf1_loss, qf2_loss, value_loss, entropy

    def learn(self, total_timesteps, loaded_step_num=0 ,callback=None,
              log_interval=4, tb_log_name="SAC", reset_num_timesteps=True, replay_wrapper=None, save_interval=0, save_path=None):

        new_tb_log = self._init_num_timesteps(reset_num_timesteps)
        callback = self._init_callback(callback)
        self.num_timesteps += loaded_step_num

        if replay_wrapper is not None:
            self.replay_buffer = replay_wrapper(self.replay_buffer)

        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name, new_tb_log) \
                as writer:

            self._setup_learn()
            if self.grad_logger:
                self.file_vars = open("./vars_logger.txt",'w')
                self.file_logger = open("./gradient_logger.txt",'w')
                self.file_mean_logger = open("./mean_gradient_logger.txt",'w')
            np.set_printoptions(threshold=sys.maxsize)

            # Transform to callable if needed
            self.learning_rate = get_schedule_fn(self.learning_rate)
            # Initial learning rate
            current_lr = self.learning_rate(1)

            start_time = time.time()
            episode_rewards = [0.0]
            episode_successes = []
            if self.action_noise is not None:
                self.action_noise.reset()
            obs = self.env.reset()
            # Retrieve unnormalized observation for saving into the buffer
            if self._vec_normalize_env is not None:
                obs_ = self._vec_normalize_env.get_original_obs().squeeze()

            n_updates = 0
            infos_values = []

            callback.on_training_start(locals(), globals())
            callback.on_rollout_start()
            nan_num = 0
            data_past = []

            for step in range(total_timesteps - loaded_step_num):
                # Before training starts, randomly sample actions
                # from a uniform distribution for better exploration.
                # Afterwards, use the learned policy
                # if random_exploration is set to 0 (normal setting)
                if self.num_timesteps < self.learning_starts or np.random.rand() < self.random_exploration:
                    # actions sampled from action space are from range specific to the environment
                    # but algorithm operates on tanh-squashed actions therefore simple scaling is used
                    unscaled_action = self.env.action_space.sample()
                    action = scale_action(self.action_space, unscaled_action)
                    weight = None
                    subgoal = None
                else:
                    #NOTE: non_subgoal
                    # action = self.policy_tf.step(obs[None], deterministic=False).flatten()
                    # weight = subgoal = None
                    # NOTE: subgoal
                    action, subgoal, weight = self.policy_tf.subgoal_step(obs[None], deterministic=False)

                    action = action.flatten()
                    unscaled_action = unscale_action(self.action_space, action)
                    # weight = self.policy_tf.get_weight(obs[None])['level1_PoseControl/weight'][0]

                    # Add noise to the action (improve exploration,
                    # not needed in general)
                    if self.action_noise is not None:
                        action = np.clip(action + self.action_noise(), -1, 1)
                    # inferred actions need to be transformed to environment action_space before stepping
                    unscaled_action = unscale_action(self.action_space, action)

                assert action.shape == self.env.action_space.shape
                
                new_obs, reward, done, info = self.env.step(unscaled_action, weight=weight, subgoal=subgoal)
                self.num_timesteps += 1

                # Only stop training if return value is False, not when it is None. This is for backwards
                # compatibility with callbacks that have no return statement.
                if callback.on_step() is False:
                    break

                # Store only the unnormalized version
                if self._vec_normalize_env is not None:
                    new_obs_ = self._vec_normalize_env.get_original_obs().squeeze()
                    reward_ = self._vec_normalize_env.get_original_reward().squeeze()
                else:
                    # Avoid changing the original ones
                    obs_, new_obs_, reward_ = obs, new_obs, reward

                # Store transition in the replay buffer.
                self.replay_buffer.add(obs_, action, reward_, new_obs_, float(done))
                obs = new_obs
                # Save the unnormalized observation
                if self._vec_normalize_env is not None:
                    obs_ = new_obs_

                # Retrieve reward and episode length if using Monitor wrapper
                maybe_ep_info = info.get('episode')
                if maybe_ep_info is not None:
                    self.ep_info_buf.extend([maybe_ep_info])

                if writer is not None:
                    # Write reward per episode to tensorboard
                    ep_reward = np.array([reward_]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    tf_util.total_episode_reward_logger(self.episode_reward, ep_reward,
                                                        ep_done, writer, self.num_timesteps)

                if self.num_timesteps % self.train_freq == 0:
                    callback.on_rollout_end()

                    mb_infos_vals = []
                    # Update policy, critics and target networks
                    for grad_step in range(self.gradient_steps):
                        # Break if the warmup phase is not over
                        # or if there are not enough samples in the replay buffer
                        if not self.replay_buffer.can_sample(self.batch_size) \
                           or self.num_timesteps < self.learning_starts:
                            break
                        n_updates += 1
                        # Compute current learning_rate
                        frac = 1.0 - step / total_timesteps
                        current_lr = self.learning_rate(frac)
                        # Update policy and critics (q functions)
                        mb_infos_vals.append(self._train_step(step, writer, current_lr))
                        # Update target network
                        if (step + grad_step) % self.target_update_interval == 0:
                            # Update target network
                            self.sess.run(self.target_update_op)
                    # Log losses and entropy, useful for monitor training
                    if len(mb_infos_vals) > 0:
                        infos_values = np.mean(mb_infos_vals, axis=0)

                    callback.on_rollout_start()

                episode_rewards[-1] += reward_
                if done:
                    if self.action_noise is not None:
                        self.action_noise.reset()
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

                    maybe_is_success = info.get('is_success')
                    if maybe_is_success is not None:
                        episode_successes.append(float(maybe_is_success))

                if len(episode_rewards[-101:-1]) == 0:
                    mean_reward = -np.inf
                else:
                    mean_reward = round(float(np.mean(episode_rewards[-101:-1])), 4)

                num_episodes = len(episode_rewards)

                if save_interval and save_path != None:
                    if (step+1) % save_interval == 0 and step:
                        print("saved")
                        self.save(save_path+"/policy_"+str(step+loaded_step_num+1), hierarchical=True)
                # Display training infos
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    fps = int(step / (time.time() - start_time))
                    logger.logkv("episodes", num_episodes)
                    logger.logkv("mean 100 episode reward", mean_reward)
                    if len(self.ep_info_buf) > 0 and len(self.ep_info_buf[0]) > 0:
                        logger.logkv('ep_rewmean', safe_mean([ep_info['r'] for ep_info in self.ep_info_buf]))
                        logger.logkv('eplenmean', safe_mean([ep_info['l'] for ep_info in self.ep_info_buf]))
                    logger.logkv("n_updates", n_updates)
                    logger.logkv("current_lr", current_lr)
                    logger.logkv("fps", fps)
                    logger.logkv('time_elapsed', int(time.time() - start_time))
                    if len(episode_successes) > 0:
                        logger.logkv("success rate", np.mean(episode_successes[-100:]))
                    if len(infos_values) > 0:
                        for (name, val) in zip(self.infos_names, infos_values):
                            logger.logkv(name, val)
                    logger.logkv("total timesteps", self.num_timesteps)
                    logger.dumpkvs()
                    # Reset infos:
                    infos_values = []
                # self.env.render()
            callback.on_training_end()
            if self.grad_logger:
                self.file_vars.close()
                self.file_logger.close()
                self.file_mean_logger.close()
            return self

    def action_probability(self, observation, state=None, mask=None, actions=None, logp=False):
        if actions is not None:
            raise ValueError("Error: SAC does not have action probabilities.")

        warnings.warn("Even though SAC has a Gaussian policy, it cannot return a distribution as it "
                      "is squashed by a tanh before being scaled and outputed.")

        return None

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions = self.policy_tf.step(observation, deterministic=deterministic)
        actions = actions.reshape((-1,) + self.action_space.shape)  # reshape to the correct action shape
        actions = unscale_action(self.action_space, actions) # scale the output for the prediction
        if not vectorized_env:
            actions = actions[0]

        return actions, None
    
    def predict_subgoal(self, observation, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        actions, subgoal, weight = self.policy_tf.subgoal_step(observation, deterministic=deterministic)
        actions = unscale_action(self.action_space, actions)

        if not vectorized_env:
            actions = actions[0]
        
        return actions, subgoal, weight

    
    def get_weight(self, observation):
        observation = np.array(observation)
        observation = observation.reshape((-1,) + self.observation_space.shape)
        return self.policy_tf.get_weight(observation)
    
    def get_primitive_action(self, observation):
        observation = np.array(observation)
        observation = observation.reshape((-1,) + self.observation_space.shape)
        return self.policy_tf.get_primitive_action(observation)
    
    def get_primitive_log_std(self, observation):
        observation = np.array(observation)
        observation = observation.reshape((-1,) + self.observation_space.shape)
        return self.policy_tf.get_primitive_log_std(observation)
    
    def get_primitive_param(self, observation):
        observation = np.array(observation)
        observation = observation.reshape((-1,) + self.observation_space.shape)
        return self.policy_tf.get_primitive_param(observation)

    def get_parameter_list(self):
        return (self.params +
                self.target_params)

    def save(self, save_path, cloudpickle=False, hierarchical=False):        
        data = {
            "composite_primitive_name": self.composite_primitive_name,
            "learning_rate": self.learning_rate,
            "buffer_size": self.buffer_size,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "tau": self.tau,
            "ent_coef": self.ent_coef if isinstance(self.ent_coef, float) else 'auto',
            "target_entropy": self.target_entropy,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "primitives": self.primitives,
            "tails": self.tails,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "n_cpu_tf_sess": self.n_cpu_tf_sess,
            "seed": self.seed,
            "action_noise": self.action_noise,
            "random_exploration": self.random_exploration,
            "_vectorize_action": self._vectorize_action,
            "policy_kwargs": self.policy_kwargs
        }

        params_to_save = self.get_parameters(hierarchical)

        self._save_to_file(save_path, data=data, params=params_to_save, cloudpickle=cloudpickle)
