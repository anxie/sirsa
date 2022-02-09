import numpy as np
from gym import spaces

from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv

np.random.seed(1)

MAX_BUFFER_SIZE = 1001
ACTION_DIM = 6
STATE_DIM = 17

CONTEXT_DIM = 8
TOTAL_TASKS = 100

TORSO_MASS_RANGE = (-0.5, 0.5)
FRICTION_RANGE = (-0.5, 0.5)

LOW = np.array([
    TORSO_MASS_RANGE[0],
    FRICTION_RANGE[0],
    *[0.0] * 6,
])
HIGH = np.array([
    TORSO_MASS_RANGE[1],
    FRICTION_RANGE[1],
    *[1.0] * 6,
])
CENTERS = np.random.uniform(
    LOW + 0.1 * (HIGH - LOW),
    HIGH - 0.1 * (HIGH - LOW),
    [TOTAL_TASKS, CONTEXT_DIM])

WIDTHS = np.random.uniform(
    0.3 * (HIGH - LOW),
    np.minimum(CENTERS - LOW, HIGH - CENTERS))

for i in range(TOTAL_TASKS):
    motor_id = np.random.choice(6)
    center = np.zeros(6)
    center[motor_id] = CENTERS[i, -6:][motor_id]

    width = np.zeros(6)
    width[motor_id] = WIDTHS[i, -6:][motor_id]
    CENTERS[i, -6:] = center.copy()
    WIDTHS[i, -6:] = width.copy()

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


class RandomizedHalfCheetahEnv(HalfCheetahEnv):
    def __init__(self,
                 observe_context=False,
                 num_tasks=1,
                 task_id=-1,
                 sample_id=-1,
                 env_type='average',
                 history_length=0,
                 history_include_actions=True,
                 history_include_states=True):

        self._observe_context = observe_context
        self._num_tasks = num_tasks
        self._task_id = task_id
        self._sample_id = sample_id
        self._env_type = env_type

        self._history_length = history_length
        self._history_include_actions = history_include_actions
        self._history_include_states = history_include_states

        self._past_actions = np.zeros((MAX_BUFFER_SIZE, ACTION_DIM))
        self._past_states = np.zeros((MAX_BUFFER_SIZE, STATE_DIM))

        self._counter = 0

        self._center = CENTERS[0]
        self._width = WIDTHS[0]
        self._context = CENTERS[0]

        super(RandomizedHalfCheetahEnv, self).__init__()

        self._torso_index = self.model.body_names.index('torso')

        self._original_mass = np.array(self.model.body_mass)
        self._original_friction = np.array(self.model.geom_friction)

    def reset(self, task_id=-1, sample_id=-1):
        if self._task_id != -1: task_id = self._task_id
        if task_id == -1: task_id = np.random.randint(self._num_tasks)

        if self._sample_id != -1: sample_id = self._sample_id

        self._center = CENTERS[task_id]
        self._width = WIDTHS[task_id]

        if self._env_type == 'worst':
            self._context = self._center + np.array([*[1] * CONTEXT_DIM]) * self._width
        elif self._env_type == 'average':
            self._context = self._center
        elif self._env_type == 'random':
            sample_id = np.random.randint(10)
            self._context = GAUSSIAN_SAMPLES[task_id, sample_id].copy()
        elif self._env_type == 'uniform':
            if sample_id == -1:
                raise ValueError('Need to pass in valid value to sample_id')
            self._context = UNIFORM_SAMPLES[task_id, sample_id].copy()
        elif self._env_type == 'gaussian':
            if sample_id == -1:
                raise ValueError('Need to pass in valid value to sample_id')
            self._context = GAUSSIAN_SAMPLES[task_id, sample_id].copy()

        self._set_parameters()

        self._past_actions = np.zeros((MAX_BUFFER_SIZE, ACTION_DIM))
        self._past_states = np.zeros((MAX_BUFFER_SIZE, STATE_DIM))
        self._counter = 0

        obs = super(RandomizedHalfCheetahEnv, self).reset()
        self._past_states[self._counter] = obs
        return self._wrap_obs(obs)

    def _set_parameters(self):
        bm = self._original_mass.copy()
        bm[self._torso_index] *= (1.0 + self._context[0])
        self.model.body_mass[:] = bm

        gf = self._original_friction.copy() * (0.5 + self._context[1])
        self.model.geom_friction[:] = gf

    def step(self, action):
        scale = np.ones(ACTION_DIM)
        motor_id = np.where(self._context[-6:] > 0)[0]
        for i in motor_id:
            scale[i] = 0 if np.random.random() < self._context[-6:][i] else 1
        scaled_action = scale * action
        next_obs, reward, done, info = super(RandomizedHalfCheetahEnv, self).step(scaled_action)

        self._past_actions[self._counter] = action
        self._counter += 1
        self._past_states[self._counter] = next_obs

        return self._wrap_obs(next_obs), reward, done, info

    def _wrap_obs(self, obs):
        curr_obs = obs
        obs = dict(observation=obs)

        if self._observe_context:
            obs.update(
                center=self._center.copy(),
                radius=self._width.copy(),
                context=self._context.copy(),
            )

        if self._history_length:
            history_states = []
            history_actions = []
            for i in range(self._history_length):
                t = max(self._counter - i - 1, 0)

                if self._history_include_states:
                    history_states.append(self._past_states[t])

                if self._history_include_actions:
                    history_actions.append(self._past_actions[t])
            t = self._counter
            obs.update(history=np.concatenate((
                [curr_obs] + history_states + history_actions
            )))

        return obs
