import numpy as np
from gym import spaces
import gym

np.random.seed(1)

MAX_BUFFER_SIZE = 51
ACTION_DIM = 1
STATE_DIM = 4

CONTEXT_DIM = 2

TOTAL_TASKS = 40

LOW = np.array([-1.0, -1.0])
HIGH = np.array([1.0, 1.0])

CENTERS = np.random.uniform(
    LOW + 0.1 * (HIGH - LOW),
    HIGH - 0.1 * (HIGH - LOW),
    [TOTAL_TASKS, CONTEXT_DIM])

WIDTHS = np.random.uniform(
    0.2 * (HIGH - LOW),
    np.minimum(CENTERS - LOW, HIGH - CENTERS))

NUM_UNIFORM_SAMPLES = 50
UNIFORM_SAMPLES = np.random.uniform(
    CENTERS - WIDTHS,
    CENTERS + WIDTHS,
    [NUM_UNIFORM_SAMPLES, TOTAL_TASKS, CONTEXT_DIM]
).transpose((1,0,2))

NUM_GAUSSIAN_SAMPLES = 50
GAUSSIAN_SAMPLES = np.random.normal(
    CENTERS,
    WIDTHS / 2.0,
    [NUM_GAUSSIAN_SAMPLES, TOTAL_TASKS, CONTEXT_DIM]
).transpose((1,0,2))

np.random.seed()


class ReacherEnv(gym.Env):
    def __init__(
        self,
        action_scale=0.01,
        boundary_dist=0.15,
        observe_context=False,
        num_tasks=1,
        task_id=-1,
        sample_id=-1,
        env_type='average',
        oracle=False,
        history_length=0,
        history_include_actions=True,
        history_include_states=True
    ):
        self._action_scale = action_scale
        self._boundary_dist = boundary_dist
        self._observe_context = observe_context
        self._num_tasks = num_tasks
        self._task_id = task_id
        self._sample_id = sample_id
        self._env_type = env_type
        self._oracle = oracle

        self._history_length = history_length
        self._history_include_actions = history_include_actions
        self._history_include_states = history_include_states

        self._past_actions = np.zeros((MAX_BUFFER_SIZE, ACTION_DIM))
        self._past_states = np.zeros((MAX_BUFFER_SIZE, STATE_DIM))

        self._counter = 0

        self._initial_position = np.array([-0.15, 0.0])
        self._target_position = np.array([+0.15, 0.0])
        self._obstacle_position = np.array([0.0, 0.0])

        u = np.ones(1)
        self.action_space = spaces.Box(-u, u, dtype='float32')

        o = self._boundary_dist * np.ones(STATE_DIM)
        obs_range = spaces.Box(-o, o, dtype='float32')

        c = self._boundary_dist * np.ones(CONTEXT_DIM)
        context_range = spaces.Box(-c, c, dtype='float32')

        items = [('observation', obs_range)]
        if self._observe_context:
            items += [
                ('center', context_range),
                ('radius', context_range),
                ('context', context_range),
            ]
        if self._history_length:
            size = 0
            if self._history_include_actions:
                size += ACTION_DIM
            if self._history_include_states:
                size += STATE_DIM
            size *= self._history_length
            hist_range = spaces.Box(np.array([-np.inf] * size), np.array([np.inf] * size), dtype='float32')
            items += [('history', hist_range)]
        self.observation_space = spaces.Dict(items)

        self.viewer = None

    def step(self, velocities):
        clipped_velocities = np.clip(velocities, a_min=-1, a_max=1) * self._action_scale
        self._position += np.array([self._x_speed, clipped_velocities[0]])
        self._position = np.clip(
            self._position,
            a_min=np.array([-self._boundary_dist, 0.0]),
            a_max=self._boundary_dist,
        )

        ob = self._get_obs()

        reward = self._compute_reward()
        info = dict(
            penalty=int(self._inside_obstacle()),
            y_position=np.abs(self._position[1]),
        )
        done = False

        self._past_actions[self._counter] = velocities
        self._counter += 1
        self._past_states[self._counter] = np.concatenate((
            self._position.copy(),
            np.array([int(self._inside_obstacle())], dtype=np.float32), 
            np.array([np.linalg.norm(self._position - self._obstacle_position)], dtype=np.float32),
        ))

        return ob, reward, done, info

    def reset(self, task_id=-1, sample_id=-1):
        self._position = self._initial_position.copy()

        if self._task_id != -1: task_id = self._task_id
        if task_id == -1: task_id = np.random.randint(self._num_tasks)

        if self._sample_id != -1: sample_id = self._sample_id

        self._center = CENTERS[task_id]
        self._width = WIDTHS[task_id]

        if self._env_type == 'worst':
            self._context = self._center + self._width
        elif self._env_type == 'average':
            self._context = self._center
        elif self._env_type == 'random':
            sample_id = np.random.randint(3)
            self._context = GAUSSIAN_SAMPLES[task_id, sample_id].copy()
        elif self._env_type == 'uniform':
            if sample_id == -1:
                raise ValueError('Need to pass in valid value to sample_id')
            self._context = UNIFORM_SAMPLES[task_id][sample_id]
        elif self._env_type == 'gaussian':
            if sample_id == -1:
                raise ValueError('Need to pass in valid value to sample_id')
            self._context = GAUSSIAN_SAMPLES[sample_id]
        else:
            raise NotImplementedError

        self._set_parameters()

        self._counter = 0

        return self._get_obs()

    def _set_parameters(self):
        self._obstacle_radius = 0.05 + 0.025 * self._context[0]
        self._x_speed = 0.008 + 0.002 * self._context[1]

    def _get_obs(self):
        obs = np.concatenate((
            self._position.copy(),
            np.array([int(self._inside_obstacle())], dtype=np.float32), 
            np.array([np.linalg.norm(self._position - self._obstacle_position)], dtype=np.float32),
        ))
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
            obs.update(history=np.concatenate((history_states + history_actions)))

        return obs

    def _distance_to_target(self):
        return np.linalg.norm(self._position - self._target_position)

    def _inside_obstacle(self):
        ratios = (self._position - self._obstacle_position) / self._obstacle_radius
        return np.sum(np.square(ratios)) < 1.0

    def _compute_reward(self):
        return 1.0 - int(self._inside_obstacle()) - 8.0 * np.abs(self._position[1])
