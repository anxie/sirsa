"""Custom Minitaur environment with target velocity.

Implements minitaur environment with rewards dependent on closeness to goal
velocity. Extends the MinitaurExtendedEnv class from PyBullet
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym import spaces

from pybullet_envs.minitaur.envs import minitaur_extended_env

ENV_DEFAULTS = {
    "accurate_motor_model_enabled": True,
    "never_terminate": False,
    "history_length": 5,
    "urdf_version": "rainbow_dash_v0",
    "history_include_actions": True,
    "control_time_step": None, #0.02,
    "control_latency": 0.0,
    "pd_latency": 0.0,
    "history_include_states": True,
    "include_leg_model": True
}

np.random.seed(1)

ACTION_DIM = 8
STATE_DIM = 12

CONTEXT_DIM = 6
TOTAL_TASKS = 100

# Relative range.
MINITAUR_BASE_MASS_ERROR_RANGE = (-1.0, 1.0)
MINITAUR_LEG_MASS_ERROR_RANGE = (-1.0, 1.0)

LOW = np.array([
    MINITAUR_BASE_MASS_ERROR_RANGE[0],
    MINITAUR_LEG_MASS_ERROR_RANGE[0],
    *[0.0] * 4,
])

HIGH = np.array([
    MINITAUR_BASE_MASS_ERROR_RANGE[1],
    MINITAUR_LEG_MASS_ERROR_RANGE[1],
    *[1.0] * 4,
])

CENTERS = np.random.uniform(
    LOW + 0.1 * (HIGH - LOW),
    HIGH - 0.1 * (HIGH - LOW),
    [TOTAL_TASKS, CONTEXT_DIM])

WIDTHS = np.random.uniform(
    0.3 * (HIGH - LOW),
    np.minimum(CENTERS - LOW, HIGH - CENTERS))

for i in range(TOTAL_TASKS):
    motor_id = np.random.choice(4)
    center = np.zeros(4)
    center[motor_id] = CENTERS[i, -4:][motor_id]

    width = np.zeros(4)
    width[motor_id] = WIDTHS[i, -4:][motor_id]
    CENTERS[i, -4:] = center.copy()
    WIDTHS[i, -4:] = width.copy()

NUM_UNIFORM_SAMPLES = 100
UNIFORM_SAMPLES = np.random.uniform(
    CENTERS - WIDTHS,
    CENTERS + WIDTHS,
    [NUM_UNIFORM_SAMPLES, TOTAL_TASKS, CONTEXT_DIM]
).transpose((1,0,2))

NUM_GAUSSIAN_SAMPLES = 100
GAUSSIAN_SAMPLES = np.random.normal(
    CENTERS,
    WIDTHS / 2.0,
    [NUM_GAUSSIAN_SAMPLES, TOTAL_TASKS, CONTEXT_DIM]
).transpose((1,0,2))

np.random.seed()


class RandomizedMinitaurEnv(minitaur_extended_env.MinitaurExtendedEnv):
    """Minitaur env with randomized parameters."""

    def __init__(self,
                 observe_context=False,
                 num_tasks=1,
                 task_id=-1,
                 sample_id=-1,
                 env_type='average',
                 oracle=False,
                 history_length=0,
                 history_include_actions=True,
                 history_include_states=True):

        self._observe_context = observe_context
        self._num_tasks = num_tasks
        self._task_id = task_id
        self._sample_id = sample_id
        self._env_type = env_type
        self._oracle = oracle

        self._history_length_h = history_length
        self._history_include_actions_h = history_include_actions
        self._history_include_states_h = history_include_states

        self.minitaur = None

        self._scale = np.ones(ACTION_DIM)

        self._task_id_copy = 0

        super(RandomizedMinitaurEnv, self).__init__(**ENV_DEFAULTS)

        self.observation_space = spaces.Box(np.array([-np.inf] * 52), np.array([np.inf] * 52), dtype='float32')
        obs_range = self.observation_space
        context_range = spaces.Box(LOW, HIGH, dtype='float32')
        items = [('observation', obs_range)]
        if self._observe_context:
            items += [
                ('center', context_range),
                ('radius', context_range),
                ('context', context_range),
            ]
        if self._history_length_h:
            size = 0
            if self._history_include_actions:
                size += ACTION_DIM
            if self._history_include_states:
                size += STATE_DIM
            size *= self._history_length_h
            size += STATE_DIM
            hist_range = spaces.Box(np.array([-np.inf] * size), np.array([np.inf] * size), dtype='float32')
            items += [('history', hist_range)]
        self.observation_space = spaces.Dict(items)

        self._original_base_masses = self.minitaur.GetBaseMassesFromURDF()
        self._original_leg_masses = self.minitaur.GetLegMassesFromURDF()

    def reset(self, task_id=-1, sample_id=-1):
        if self._task_id != -1: task_id = self._task_id
        if task_id == -1: task_id = np.random.randint(self._num_tasks)

        if self._sample_id != -1: sample_id = self._sample_id

        self._center = CENTERS[task_id]
        self._width = WIDTHS[task_id]
        self._task_id_copy = task_id

        if self._env_type == 'worst':
            self._context = self._center + np.array([1, 1, 1, 1, 1, 1]) * self._width
        elif self._env_type == 'average':
            self._context = self._center
        elif self._env_type == 'random':
            sample_id = np.random.randint(10)
            self._context = GAUSSIAN_SAMPLES[task_id, sample_id].copy()
        elif self._env_type == 'uniform':
            if sample_id == -1:
                self._context = None
            self._context = UNIFORM_SAMPLES[task_id, sample_id].copy()
        elif self._env_type == 'gaussian':
            if sample_id == -1:
                self._context = None
            self._context = GAUSSIAN_SAMPLES[task_id, sample_id].copy()

        if self._oracle:
            self._center = self._context.copy()

        self._set_parameters()

        obs = super(RandomizedMinitaurEnv, self).reset()
        return self._wrap_obs(obs)

    def _set_parameters(self):
        if self.minitaur:
            base_masses = [(1.0 + 0.2 * self._context[0]) * m for m in self._original_base_masses]
            self.minitaur.SetBaseMasses(base_masses)
            leg_masses = [(1.0 + 0.2 * self._context[1]) * m for m in self._original_leg_masses]
            self.minitaur.SetLegMasses(leg_masses)

    def step(self, action):
        scale = np.ones(ACTION_DIM)
        motor_id = np.where(self._context[-4:] > 0)[0]
        for i in motor_id:
            p = np.random.random()
            scale[i * 2:(i + 1) * 2] = 0 if p < self._context[-4:][i] else 1

        scaled_action = scale * action
        next_obs, reward, done, info = super(RandomizedMinitaurEnv, self).step(scaled_action)

        return self._wrap_obs(next_obs), reward, done, info

    def _wrap_obs(self, obs):
        obs = dict(observation=obs)

        mean = np.zeros(CONTEXT_DIM)
        mean[-4:] = 0.5

        if self._observe_context:
            obs.update(
                center=self._center.copy() - mean,
                radius=self._width.copy(),
                context=self._context.copy() - mean,
            )

        if self._history_length_h:
            history_states = []
            history_actions = []
            for i in range(self._history_length_h):
                t = max(self._counter - i - 1, 0)

                if self._history_include_states_h:
                    history_states.append(self._past_parent_observations[t])

                if self._history_include_actions_h:
                    history_actions.append(self._past_actions[t])
            t = self._counter

            curr_obs = obs['observation'][:STATE_DIM]
            obs.update(history=np.concatenate((
                [curr_obs] + history_states + history_actions
            )))

        if self._history_length:
            history_states = []
            history_actions = []
            for i in range(self._history_length):
                t = max(self._counter - i - 1, 0)

                if self._history_include_states:
                    history_states.append(self._past_parent_observations[t])

                if self._history_include_actions:
                    history_actions.append(self._past_actions[t])
            t = self._counter

            curr_obs = obs['observation'][:STATE_DIM]
            obs.update(observation=np.concatenate((
                [curr_obs] + history_actions
            )))

        return obs
