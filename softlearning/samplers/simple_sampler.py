from collections import defaultdict, OrderedDict

import tensorflow as tf
import numpy as np
from flatten_dict import flatten, unflatten

from softlearning.models.utils import flatten_input_structure
from .base_sampler import BaseSampler
from softlearning.replay_pools.percentile_replay_pool import MultitaskPercentileReplayPool


class SimpleSampler(BaseSampler):
    def __init__(self, latent_dim=0, update_belief=False, **kwargs):
        super(SimpleSampler, self).__init__(**kwargs)

        self._path_length = 0
        self._path_return = 0
        self._current_path = defaultdict(list)
        self._last_path_return = 0
        self._max_path_return = -np.inf
        self._n_episodes = 0
        self._task_id = 0
        self._sample_id = 0
        self._current_observation = None
        self._total_samples = 0

        self._encode_history = latent_dim > 0
        self._latent_dim = latent_dim
        if self._encode_history:
            self._latent_context = np.zeros(self._latent_dim * 2)[None, ...]
        self._update_belief = update_belief
        self._samples = None

    def attach(self, algorithm):
        self.alg = algorithm

    def _policy_input(self, sample=None):
        observation = {
            key: self._current_observation[key][None, ...]
            for key in self.policy.observation_keys
        }
        if self._encode_history:
            if self._update_belief and np.any(self._latent_context != 0):
                latent_center = self._latent_context[:, :self._latent_dim]
                latent_radius = self._latent_context[:, self._latent_dim:]
            else:
                latent_center = self._current_observation['center'][None, ...]
                latent_radius = self._current_observation['radius'][None, ...]
            self._latent_context = self.alg._session.run(self.alg._latent_context,
                feed_dict={
                    self.alg._placeholders['observations']['history']: self._current_observation['history'][None, ...],
                    self.alg._placeholders['observations']['center']: latent_center,
                    self.alg._placeholders['observations']['radius']: latent_radius,
                })
            self._sum_of_squared_error = np.sum(np.square(self._current_observation['context'] - self._latent_context[0, :self._latent_dim]))
            observation.update({
                'center': self._latent_context[:, :self._latent_dim],
                'radius': self._latent_context[:, self._latent_dim:],
            })

        if sample is not None:
            observation.update({'context': sample[None, ...]})
        observation = flatten_input_structure(observation)

        return observation

    def _process_sample(self,
                        observation,
                        action,
                        reward,
                        terminal,
                        next_observation,
                        info):
        processed_observation = {
            'observations': observation,
            'actions': action,
            'rewards': [reward],
            'terminals': [terminal],
            'next_observations': next_observation,
            'infos': info,
        }

        return processed_observation

    def sample(self, task_id=-1, sample_id=-1):
        if self._current_observation is None:
            if task_id != -1: self._task_id = task_id
            if sample_id != -1: self._sample_id = sample_id
            self._current_observation = self.env.reset(self._task_id, self._sample_id)

        action = self.policy.actions_np(self._policy_input())[0]

        next_observation, reward, terminal, info = self.env.step(action)
        if self._encode_history:
            info['sum_of_squared_error'] = self._sum_of_squared_error
        self._path_length += 1
        self._path_return += reward
        self._total_samples += 1

        processed_sample = self._process_sample(
            observation=self._current_observation,
            action=action,
            reward=reward,
            terminal=terminal,
            next_observation=next_observation,
            info=info,
        )

        for key, value in flatten(processed_sample).items():
            self._current_path[key].append(value)

        if terminal or self._path_length >= self._max_path_length:
            last_path = unflatten({
                field_name: np.array(values)
                for field_name, values in self._current_path.items()
            })

            self.pool.add_path({
                key: value
                for key, value in last_path.items()
                if key != 'infos'
            })

            self._last_n_paths.appendleft(last_path)

            self._max_path_return = max(self._max_path_return,
                                        self._path_return)
            self._last_path_return = self._path_return

            self.policy.reset()
            self.pool.terminate_episode()
            self._current_observation = None
            self._latent_context = np.zeros(self._latent_dim * 2)[None, ...]
            self._path_length = 0
            self._path_return = 0
            self._current_path = defaultdict(list)

            self._n_episodes += 1

            if isinstance(self.pool, MultitaskPercentileReplayPool):
                if task_id == -1:
                    self._task_id = self._n_episodes % self.pool._num_tasks
                self.pool.set_task(self._task_id)
        else:
            self._current_observation = next_observation

        return next_observation, reward, terminal, info

    def random_batch(self, batch_size=None, **kwargs):
        batch_size = batch_size or self._batch_size
        # observation_keys = getattr(self.env, 'observation_keys', None)

        return self.pool.random_batch(batch_size, **kwargs)

    def get_diagnostics(self):
        diagnostics = super(SimpleSampler, self).get_diagnostics()
        diagnostics.update({
            'max-path-return': self._max_path_return,
            'last-path-return': self._last_path_return,
            'episodes': self._n_episodes,
            'total-samples': self._total_samples,
        })
        if self._encode_history:
            diagnostics.update({
                'sum-of-squared-error': self._sum_of_squared_error,
            })

        return diagnostics
