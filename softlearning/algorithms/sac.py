from collections import OrderedDict
from numbers import Number

import numpy as np
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)

import tensorflow_probability as tfp
tfd = tfp.distributions
from flatten_dict import flatten

import tensorflow_probability as tfp
tfd = tfp.distributions

from softlearning.models.utils import flatten_input_structure
from .rl_algorithm import RLAlgorithm

from softlearning.models.feedforward import feedforward_model


def td_target(reward, discount, next_value):
    return reward + discount * next_value


class SAC(RLAlgorithm):
    """Soft Actor-Critic (SAC)

    References
    ----------
    [1] Tuomas Haarnoja*, Aurick Zhou*, Kristian Hartikainen*, George Tucker,
        Sehoon Ha, Jie Tan, Vikash Kumar, Henry Zhu, Abhishek Gupta, Pieter
        Abbeel, and Sergey Levine. Soft Actor-Critic Algorithms and
        Applications. arXiv preprint arXiv:1812.05905. 2018.
    """

    def __init__(
            self,
            training_environment,
            evaluation_environment,
            policy,
            Qs,
            pool,
            plotter=None,

            lr=3e-4,
            reward_scale=1.0,
            target_entropy='auto',
            discount=0.99,
            tau=5e-3,
            target_update_interval=1,
            action_prior='uniform',
            reparameterize=False,

            save_full_state=False,

            cvar_samples=50,
            cvar_alpha=0.,
            threshold_iterations=0,

            latent_dim=0,
            state_dim=0,
            pretrain_iterations=0,
            ensemble_size=1,
            encoder_size=(64, 64),
            history_length=0,
            predict_context=False,

            redq_subset_size=2,
            num_Qs=2,

            **kwargs,
    ):
        """
        Args:
            env (`SoftlearningEnv`): Environment used for training.
            policy: A policy function approximator.
            initial_exploration_policy: ('Policy'): A policy that we use
                for initial exploration which is not trained by the algorithm.
            Qs: Q-function approximators. The min of these
                approximators will be used. Usage of at least two Q-functions
                improves performance by reducing overestimation bias.
            pool (`PoolBase`): Replay pool to add gathered samples to.
            plotter (`QFPolicyPlotter`): Plotter instance to be used for
                visualizing Q-function during training.
            lr (`float`): Learning rate used for the function approximators.
            discount (`float`): Discount factor for Q-function updates.
            tau (`float`): Soft value function target update weight.
            target_update_interval ('int'): Frequency at which target network
                updates occur in iterations.
            reparameterize ('bool'): If True, we use a gradient estimator for
                the policy derived using the reparameterization trick. We use
                a likelihood ratio based estimator otherwise.
        """

        super(SAC, self).__init__(**kwargs)

        self._training_environment = training_environment
        self._evaluation_environment = evaluation_environment
        self._policy = policy

        self._Qs = Qs
        self._Q_targets = tuple(tf.keras.models.clone_model(Q) for Q in Qs)

        self._pool = pool
        self._plotter = plotter

        self._policy_lr = lr
        self._Q_lr = lr

        self._reward_scale = reward_scale
        self._target_entropy = (
            -np.prod(self._training_environment.action_space.shape)
            if target_entropy == 'auto'
            else target_entropy)

        self._discount = discount
        self._tau = tau
        self._target_update_interval = target_update_interval
        self._action_prior = action_prior

        self._reparameterize = reparameterize

        self._save_full_state = save_full_state

        self._cvar_samples = cvar_samples
        self._cvar_alpha = cvar_alpha
        self._threshold_iterations = threshold_iterations

        if 'center' in training_environment.observation_space.spaces:
            self._context_dim = training_environment.observation_space['center'].shape[0]

        self._batch_size = 256
        self._per_task_batch_size = int(self._batch_size / self._pool.task_batch_size)

        self._encode_history = latent_dim > 0
        if self._encode_history:
            self._latent_dim = latent_dim
            self._state_dim = state_dim
            self._pretrain_iterations = pretrain_iterations
            self._ensemble_size = ensemble_size
            self._encoder_size = encoder_size

            self._history_length = history_length
            self._predict_context = predict_context

        self._redq_subset_size = redq_subset_size
        self._num_Qs = num_Qs

        self._build()

    def _build(self):
        super(SAC, self)._build()
        if self._encode_history:
            self._init_encoder_update()
        self._init_actor_update()
        self._init_critic_update()
        self._init_diagnostics_ops()

    def _process_input(self, input_dict):
        if self._encode_history:
            input_dict.update({
                'center': tf.stop_gradient(self._latent_context[:, :self._context_dim]),
            })
            if 'radius' in input_dict:
                input_dict.update({
                    'radius': tf.stop_gradient(self._latent_context[:, self._context_dim:]),
                })
        return input_dict

    def _get_Q_target(self):
        policy_inputs = self._process_input({
            name: self._placeholders['next_observations'][name]
            for name in self._policy.observation_keys
        })
        policy_inputs = flatten_input_structure(policy_inputs)
        next_actions = self._policy.actions(policy_inputs)
        next_log_pis = self._policy.log_pis(policy_inputs, next_actions)

        next_Q_observations = {
            name: self._placeholders['next_observations'][name]
            for name in self._Qs[0].observation_keys
        }
        next_Q_inputs = flatten_input_structure(
            {**next_Q_observations, 'actions': next_actions})
        next_Qs_values = tuple(Q(next_Q_inputs) for Q in self._Q_targets)

        if self._redq_subset_size < self._num_Qs:
            redq_idxs = tf.range(len(self._Qs))
            redq_ridxs = tf.random.shuffle(redq_idxs)[:self._redq_subset_size]
            next_Qs_values_subset = tf.gather(next_Qs_values, redq_ridxs, axis=0)
            min_next_Q = tf.reduce_min(next_Qs_values_subset, axis=0)
        else:
            min_next_Q = tf.reduce_min(next_Qs_values, axis=0)
        next_values = min_next_Q - self._alpha * next_log_pis

        terminals = tf.cast(self._placeholders['terminals'], next_values.dtype)

        Q_target = td_target(
            reward=self._reward_scale * self._placeholders['rewards'],
            discount=self._discount,
            next_value=(1 - terminals) * next_values)

        return tf.stop_gradient(Q_target)

    def _init_critic_update(self):
        """Create minimization operation for critic Q-function.

        Creates a `tf.optimizer.minimize` operation for updating
        critic Q-function with gradient descent, and appends it to
        `self._training_ops` attribute.

        See Equations (5, 6) in [1], for further information of the
        Q-function update rule.
        """
        Q_target = self._get_Q_target()
        assert Q_target.shape.as_list() == [None, 1]

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_inputs = flatten_input_structure({
            **Q_observations, 'actions': self._placeholders['actions']})

        Q_values = self._Q_values = tuple(Q(Q_inputs) for Q in self._Qs)

        Q_losses = self._Q_losses = tuple(
            tf.compat.v1.losses.mean_squared_error(
                labels=Q_target, predictions=Q_value, weights=0.5)
            for Q_value in Q_values)

        self._Q_optimizers = tuple(
            tf.compat.v1.train.AdamOptimizer(
                learning_rate=self._Q_lr,
                name='{}_{}_optimizer'.format(Q._name, i)
            ) for i, Q in enumerate(self._Qs))

        Q_training_ops = tuple(
            Q_optimizer.minimize(loss=Q_loss, var_list=Q.trainable_variables)
            for i, (Q, Q_loss, Q_optimizer)
            in enumerate(zip(self._Qs, Q_losses, self._Q_optimizers)))

        self._training_ops.update({'Q': tf.group(Q_training_ops)})

    def _init_actor_update(self):
        """Create minimization operations for policy and entropy.

        Creates a `tf.optimizer.minimize` operations for updating
        policy and entropy with gradient descent, and adds them to
        `self._training_ops` attribute.

        See Section 4.2 in [1], for further information of the policy update,
        and Section 5 in [1] for further information of the entropy update.
        """

        policy_inputs = self._process_input({
            name: self._placeholders['observations'][name]
            for name in self._policy.observation_keys
        })
        policy_inputs = flatten_input_structure(policy_inputs)
        actions = self._policy.actions(policy_inputs)
        log_pis = self._policy.log_pis(policy_inputs, actions)

        assert log_pis.shape.as_list() == [None, 1]

        log_alpha = self._log_alpha = tf.compat.v1.get_variable(
            'log_alpha',
            dtype=tf.float32,
            initializer=0.0)
        alpha = tf.exp(log_alpha)

        if isinstance(self._target_entropy, Number):
            alpha_loss = -tf.reduce_mean(
                log_alpha * tf.stop_gradient(log_pis + self._target_entropy))

            self._alpha_optimizer = tf.compat.v1.train.AdamOptimizer(
                self._policy_lr, name='alpha_optimizer')
            self._alpha_train_op = self._alpha_optimizer.minimize(
                loss=alpha_loss, var_list=[log_alpha])

            self._training_ops.update({
                'temperature_alpha': self._alpha_train_op
            })

        self._alpha = alpha

        if self._action_prior == 'normal':
            policy_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.zeros(self._action_shape),
                scale_diag=tf.ones(self._action_shape))
            policy_prior_log_probs = policy_prior.log_prob(actions)
        elif self._action_prior == 'uniform':
            policy_prior_log_probs = 0.0

        Q_observations = {
            name: self._placeholders['observations'][name]
            for name in self._Qs[0].observation_keys
        }
        Q_inputs = flatten_input_structure({
            **Q_observations, 'actions': actions})
        Q_log_targets = tuple(Q(Q_inputs) for Q in self._Qs)

        if self._redq_subset_size < self._num_Qs:
            redq_idxs = tf.range(len(self._Qs))
            redq_ridxs = tf.random.shuffle(redq_idxs)[:self._redq_subset_size]
            Q_log_targets_subset = tf.gather(Q_log_targets, redq_ridxs, axis=0)
            min_Q_log_target = tf.reduce_min(Q_log_targets_subset, axis=0)
        else:
            min_Q_log_target = tf.reduce_min(Q_log_targets, axis=0)

        if self._reparameterize:
            policy_kl_losses = (
                alpha * log_pis
                - min_Q_log_target
                - policy_prior_log_probs)
        else:
            raise NotImplementedError

        assert policy_kl_losses.shape.as_list() == [None, 1]

        self._policy_losses = policy_kl_losses
        policy_loss = tf.reduce_mean(policy_kl_losses)

        self._policy_optimizer = tf.compat.v1.train.AdamOptimizer(
            learning_rate=self._policy_lr,
            name="policy_optimizer")

        policy_train_op = self._policy_optimizer.minimize(
            loss=policy_loss,
            var_list=self._policy.trainable_variables)

        self._training_ops.update({'policy_train_op': policy_train_op})

        if self._cvar_alpha > 0:
            if self._encode_history:
                center = tf.stop_gradient(self._latent_context[:, :self._context_dim])
                radius = tf.stop_gradient(self._latent_context[:, self._context_dim:])
            else:
                center = self._placeholders['observations']['center']
                radius = self._placeholders['observations']['radius']

            samples = [tf.random.uniform(
                shape=tf.shape(center),
                minval=center - radius,
                maxval=center + radius,
            ) for _ in range(self._cvar_samples)]

            Q_inputs = [flatten_input_structure({
                'observations': self._placeholders['observations']['observation'],
                'center': sample,
                'actions': actions}) for sample in samples]
            Q_log_targets = [tuple(Q(inputs) for Q in self._Qs) for inputs in Q_inputs]
            if self._redq_subset_size < self._num_Qs:
                Q_log_targets_subset = [tf.gather(target, redq_ridxs, axis=0) for target in Q_log_targets]
                min_Q_log_target = [tf.reduce_min(target, axis=0) for target in Q_log_targets_subset]
            else:
                min_Q_log_target = [tf.reduce_min(target, axis=0) for target in Q_log_targets]
            sorted = tf.sort(min_Q_log_target, axis=0)
            cvar_Q_log_target = tf.reduce_mean(sorted[:int(self._cvar_alpha * self._cvar_samples)], axis=0)

            policy_kl_losses = (
                alpha * log_pis
                - cvar_Q_log_target
                - policy_prior_log_probs)

            self._robust_policy_losses = policy_kl_losses

            self._robust_policy_train_op = self._policy_optimizer.minimize(
                loss=tf.reduce_mean(policy_kl_losses),
                var_list=self._policy.trainable_variables)

    def _parse_history(self, history):
        state_dim = self._state_dim or self._training_environment.observation_space.spaces['observation'].shape[0]
        action_dim = self._training_environment.action_space.shape[0]
        states = tf.reshape(
            history[:, :(self._history_length + 1) * state_dim],
            [-1, self._history_length + 1, state_dim])
        actions = tf.reshape(
            history[:, -self._history_length * action_dim:],
            [-1, self._history_length, action_dim])
        return states[:, :-1], actions, states[:, 1:]

    def _init_encoder_update(self):
        """Create minimization operation for the system identification model."""

        # note: ensemble size needs to divide into batch size
        center = self._placeholders['observations']['center']
        radius = self._placeholders['observations']['radius']

        next_states, actions, states = self._parse_history(self._placeholders['observations']['history'])

        encoder_inputs_h = []
        for i in range(self._history_length):
            encoder_inputs = tf.concat([
                states[:, i],
                actions[:, i],
                next_states[:, i],
                center,
                radius,
            ], axis=-1)
            encoder_inputs_h.append(encoder_inputs)

        # encode data
        self._encoders = [feedforward_model(
            hidden_layer_sizes=self._encoder_size,
            output_size=self._latent_dim,
            name='encoder{}'.format(i))
            for i in range(self._ensemble_size)]

        latent_contexts_h = []
        for i in range(self._history_length):
            latent_contexts = [
                center + tf.clip_by_value(encoder(encoder_inputs_h[i]), -radius, radius)
                for encoder in self._encoders]
            latent_contexts_h.append(latent_contexts)

        latent_contexts = tf.reduce_mean(tf.stack(latent_contexts_h, axis=0), axis=0)

        if self._ensemble_size > 1:
            self._latent_mean = latent_mean = tf.reduce_mean(tf.stack(latent_contexts, axis=0), axis=0)
            self._latent_std = latent_std = tf.math.reduce_std(tf.stack(latent_contexts, axis=0), axis=0)
            self._latent_context = tf.concat([latent_mean, latent_std], axis=-1)
        else:
            self._latent_context = tf.concat([latent_contexts[0], tf.zeros_like(latent_contexts[0])], axis=-1)

        # predict true context
        if self._predict_context:
            true_context = self._placeholders['observations']['context']
            mini_batch_size = self._batch_size // self._ensemble_size
            encoder_losses = [tf.reduce_sum(
                tf.square(latent_contexts[i] - true_context)[i * mini_batch_size:(i+1) * mini_batch_size, :], axis=-1)
                for i in range(self._ensemble_size)]

            self._encoder_losses = tf.reduce_sum(encoder_losses) / self._batch_size

            self._encoder_optimizers = tuple(
                tf.compat.v1.train.AdamOptimizer(
                    learning_rate=1e-3,
                    name='encoder{}_optimizer'.format(i)
                ) for i in range(self._ensemble_size))

            encoder_train_ops = tuple(
                encoder_optimizer.minimize(loss=self._encoder_losses, var_list=encoder.trainable_variables)
                for encoder_optimizer, encoder_loss, encoder
                in zip(self._encoder_optimizers, encoder_losses, self._encoders))

            self._training_ops.update({'encoder_train_op': tf.group(encoder_train_ops)})

    def _init_diagnostics_ops(self):
        diagnosables = OrderedDict((
            ('Q_value', self._Q_values),
            ('Q_loss', self._Q_losses),
            ('policy_loss', self._policy_losses),
            ('alpha', self._alpha)
        ))
        if self._encode_history and self._predict_context:
            diagnosables.update({
                'encoder_loss': self._encoder_losses,
            })
        if self._cvar_alpha > 0:
            diagnosables.update({
                'robust_policy_loss': self._robust_policy_losses,
            })

        diagnostic_metrics = OrderedDict((
            ('mean', tf.reduce_mean),
            ('std', lambda x: tfp.stats.stddev(x, sample_axis=None)),
        ))

        self._diagnostics_ops = OrderedDict([
            (f'{key}-{metric_name}', metric_fn(values))
            for key, values in diagnosables.items()
            for metric_name, metric_fn in diagnostic_metrics.items()
        ])

    def _init_training(self):
        self._update_target(tau=1.0)

    def _update_target(self, tau=None):
        tau = tau or self._tau

        for Q, Q_target in zip(self._Qs, self._Q_targets):
            source_params = Q.get_weights()
            target_params = Q_target.get_weights()
            Q_target.set_weights([
                tau * source + (1.0 - tau) * target
                for source, target in zip(source_params, target_params)
            ])

    def _do_training(self, iteration, batch):
        """Runs the operations for updating training and target ops."""

        feed_dict = self._get_feed_dict(iteration, batch)
        if self._encode_history and iteration < self._pretrain_iterations:
            self._session.run(self._training_ops['encoder_train_op'], feed_dict)
        else:
            self._session.run(self._training_ops, feed_dict)

        if self._cvar_alpha > 0 and iteration >= self._threshold_iterations:
            self._training_ops.update({
                'policy_train_op': self._robust_policy_train_op
            })

        if iteration % self._target_update_interval == 0:
            # Run target ops here.
            self._update_target()

    def _get_feed_dict(self, iteration, batch):
        """Construct a TensorFlow feed dictionary from a sample batch."""

        batch_flat = flatten(batch)
        placeholders_flat = flatten(self._placeholders)

        feed_dict = {
            placeholders_flat[key]: batch_flat[key]
            for key in placeholders_flat.keys()
            if key in batch_flat.keys()
        }

        if iteration is not None:
            feed_dict[self._placeholders['iteration']] = iteration

        return feed_dict

    def get_diagnostics(self,
                        iteration,
                        batch,
                        training_paths,
                        evaluation_paths):
        """Return diagnostic information as ordered dictionary.

        Also calls the `draw` method of the plotter, if plotter defined.
        """

        feed_dict = self._get_feed_dict(iteration, batch)
        # TODO(hartikainen): We need to unwrap self._diagnostics_ops from its
        # tensorflow `_DictWrapper`.
        diagnostics = self._session.run({**self._diagnostics_ops}, feed_dict)

        observations = {
            name: batch['observations'][name]
            for name in self._policy.observation_keys
        }
        if self._encode_history:
            latent_context = self._session.run(self._latent_context, feed_dict)
            observations.update({
                'center': latent_context[:, :self._context_dim],
                'radius': latent_context[:, self._context_dim:],
            })
        policy_inputs = flatten_input_structure(observations)

        diagnostics.update(OrderedDict([
            (f'policy/{key}', value)
            for key, value in
            self._policy.get_diagnostics(policy_inputs).items()
        ]))

        if self._plotter:
            self._plotter.draw()

        return diagnostics

    @property
    def tf_saveables(self):
        saveables = {
            '_policy_optimizer': self._policy_optimizer,
            **{
                f'Q_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._Q_optimizers)
            },
            '_log_alpha': self._log_alpha,
        }

        if hasattr(self, '_alpha_optimizer'):
            saveables['_alpha_optimizer'] = self._alpha_optimizer

        if hasattr(self, '_encoder_optimizers'):
            saveables.update({
                f'encoder_optimizer_{i}': optimizer
                for i, optimizer in enumerate(self._encoder_optimizers)
            })

        return saveables
