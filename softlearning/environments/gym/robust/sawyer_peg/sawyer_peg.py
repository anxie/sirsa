import numpy as np
from gym import spaces
from scipy.spatial.transform import Rotation
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
from softlearning.environments.gym.robust.sawyer_reacher.sawyer_reacher import SawyerReachingEnv

np.random.seed(1)

MAX_BUFFER_SIZE = 1001
ACTION_DIM = 7
STATE_DIM = 17

CONTEXT_DIM = 2
TOTAL_TASKS = 100

LOW = np.array([-0.5, -0.5,])
HIGH = np.array([0.5, 0.5,])
CENTERS = np.random.uniform(
    LOW + 0.1 * (HIGH - LOW),
    HIGH - 0.1 * (HIGH - LOW),
    [TOTAL_TASKS, CONTEXT_DIM])

WIDTHS = np.random.uniform(
    0.1 * (HIGH - LOW),
    np.minimum(CENTERS - LOW, HIGH - CENTERS))

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


def quat2euler(quat):
    rot = Rotation.from_quat(np.array(quat))
    rot_euler = rot.as_euler()
    return rot_euler


def euler2quat(yaw, pitch, roll):
    rot = Rotation.from_euler('zyx', [yaw, pitch, roll], degrees=False)
    rot_quat = rot.as_quat()
    return rot_quat


class SawyerPegInsertionEnv(SawyerReachingEnv):
    """Inserting a peg into a box (which is at a fixed location)"""

    def __init__(self,
                 xml_path=None,
                 goal_site_name=None,
                 box_site_name=None,
                 action_mode='joint_delta_position',
                 *args,
                 **kwargs):

        self.body_id_box = 0
        if xml_path is None:
            xml_path = os.path.join(SCRIPT_DIR, 'assets/sawyer_peg_insertion.xml')
        if goal_site_name is None:
            goal_site_name = 'goal_insert_site'
        super(SawyerPegInsertionEnv, self).__init__(
            xml_path=xml_path,
            goal_site_name=goal_site_name,
            action_mode=action_mode,
            *args,
            **kwargs)

        if box_site_name is None:
            box_site_name = "box"
        self.body_id_box = self.model.body_name2id(box_site_name)

    def reset_model(self):
        angles = self.init_qpos.copy()
        velocities = self.init_qvel.copy()
        self.set_state(angles, velocities) #this sets qpos and qvel + calls sim.forward
        return self.get_obs()

    def reset(self):
        # reset task (this is a single-task case)
        self.model.body_pos[self.body_id_box] = np.array([0.5, 0, 0])
        # original mujoco reset
        self.sim.reset()
        ob = self.reset_model()
        return ob


class RandomizedSawyerPegInsertion2BoxEnv(SawyerPegInsertionEnv):
    """
    Inserting a peg into a box (which could be in various places).
    This env is the multi-task version of peg insertion. The reward always gets concatenated to obs.
    """

    def __init__(self,
                 observe_context=False,
                 num_tasks=1,
                 task_id=-1,
                 sample_id=-1,
                 env_type='average',
                 history_length=0,
                 history_include_actions=True,
                 history_include_states=True,
                 xml_path=None,
                 goal_site_name=None,
                 box_site_name=None,
                 action_mode='joint_delta_position',
                 *args,
                 **kwargs):

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

        self._context = self._center.copy()

        self._correct_peg_id = 0

        self._action_scale = 1.0

        if xml_path is None:
            xml_path = os.path.join(SCRIPT_DIR, 'assets/sawyer_peg_insertion_2box.xml')
        if goal_site_name is None:
            goal_site_name = 'goal_insert_site1'
        if box_site_name is None:
            box_site_name = "box1"

        self.site_id_goals = []

        super(RandomizedSawyerPegInsertion2BoxEnv, self).__init__(
            xml_path=xml_path,
            goal_site_name=goal_site_name,
            box_site_name=box_site_name,
            action_mode=action_mode,
            *args,
            **kwargs)

        self._peg_geom_id = self.model.geom_name2id('peg')
        self._original_peg_size = self.model.geom_size[self._peg_geom_id].copy()

        self.site_id_goals = [self.model.site_name2id('goal_insert_site1'),
                              self.model.site_name2id('goal_insert_site2'),]
        self.body_id_box1 = self.model.body_name2id("box1")
        self.body_id_box2 = self.model.body_name2id("box2")

        obs_range = self.observation_space
        context_range = spaces.Box(LOW, HIGH, dtype='float32')
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

    def reset(self, task_id=-1, sample_id=-1):
        if self._task_id != -1: task_id = self._task_id
        if task_id == -1: task_id = np.random.randint(self._num_tasks)

        if self._sample_id != -1: sample_id = self._sample_id

        self._center = CENTERS[task_id]
        self._width = WIDTHS[task_id]

        if self._env_type == 'worst':
            self._context = self._center + self._width
        elif self._env_type == 'average':
            self._context = self._center.copy()
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

        self.sim.reset()
        obs = self.reset_model()
        self._past_states[self._counter] = obs
        return self._wrap_obs(obs)

    def _set_parameters(self, context=None):
        if context is not None:
            self._context = context.copy()

        size = 0.01 * (self._context[0] + 0.25)
        self.model.geom_size[self._peg_geom_id] = self._original_peg_size + np.array([size, size, 0])
        if self._context[0] < 0:    # smaller peg
            self._correct_peg_id = 1    # smaller, closer box
        else:
            self._correct_peg_id = 0    # larger, farther box

        self._action_scale = 1.0 + self._context[1]

    def get_obs_dim(self):
        return len(self.get_obs())

    def step(self, action):
        self.do_step(action * self._action_scale)
        obs = self.get_obs()
        rewards, scores, sparse_rewards = [], [], []

        if self.site_id_goals:
            for site in [self.body_id_box1, self.body_id_box2]:
                reward, score, sparse_reward = self.compute_reward(get_score=True, goal_id_override=site)
                rewards.append(reward)
                scores.append(score)
                sparse_rewards.append(sparse_reward)

        else:
            rewards = scores = sparse_rewards = [0, 0]

        if self._correct_peg_id == 0 and scores[0] > scores[1] or self._correct_peg_id == 1 and scores[1] > scores[0]:
            correct = 1
        else:
            correct = 0

        done = False
        info = dict(
            score_0=scores[0],
            score_1=scores[1],
            sparse_reward_0=sparse_rewards[0],
            sparse_reward_1=sparse_rewards[1],
            success_0=int(sparse_rewards[0] > 0),
            success_1=int(sparse_rewards[1] > 0),
            success=int(np.any(np.array(sparse_rewards) > 0)),
            correct=correct,
        )  # can populate with more info, as desired, for tb logging

        self._past_actions[self._counter] = action
        self._counter += 1
        self._past_states[self._counter] = obs

        reward = np.sum(sparse_rewards)
        return self._wrap_obs(obs), reward, done, info

    def _wrap_obs(self, obs):
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
