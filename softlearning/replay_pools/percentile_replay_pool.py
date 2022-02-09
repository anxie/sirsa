import numpy as np
from flatten_dict import flatten, unflatten
import math

from .simple_replay_pool import SimpleReplayPool
from .flexible_replay_pool import Field


class PercentileReplayPool(object):
    def __init__(self,
                 environment,
                 percentile,
                 *args,
                 **kwargs):
        extra_fields = {
            'returns': Field(
                name='returns',
                dtype='float32',
                shape=(1, )),
        }
        self._pool = SimpleReplayPool(
            environment=environment,
            *args,
            **kwargs
        )
        self._pool2 = SimpleReplayPool(
            environment=environment,
            *args,
            extra_fields=extra_fields,
            **kwargs
        )
        self._percentile = percentile

    def terminate_episode(self, stage=0):
        self._pool.terminate_episode()
        if stage > 0: self._pool2.terminate_episode()

    @property
    def size(self):
        return self._pool.size

    def add_path(self, path, stage=0):
        self._pool.add_path(path)
        if stage > 0:
            ret = path['rewards'].sum()
            ret = np.repeat(ret, path['rewards'].shape[0]).reshape(-1, 1)
            path.update({'returns': ret})
            self._pool2.add_path(path)

    def filter_batch(self, batch, indices):
        batch_flat = flatten(batch)
        field_names_flat = self._pool.fields_flat.keys()
        filtered_batch_flat = {
            field_name: batch_flat[field_name][indices]
            for field_name in field_names_flat
        }
        filtered_batch = unflatten(filtered_batch_flat)
        return filtered_batch

    def random_batch(self, batch_size, stage=0):
        if stage > 0 and self._percentile < 1:
            new_batch_size = int(batch_size / self._percentile)
            if self._pool2.size < new_batch_size:
                batch = self._pool.random_batch(batch_size)
            else:
                batch = self._pool2.random_batch(new_batch_size)
                # batch = self._pool2.last_n_batch(new_batch_size)
                threshold = np.percentile(batch['returns'], self._percentile * 100)
                indices = [
                    i for i, ret in enumerate(batch['returns'])
                    if ret <= threshold]
                batch = self.filter_batch(batch, indices)
        else:
            batch = self._pool.random_batch(batch_size)

        return batch

    def save_latest_experience(self, pickle_path):
        self._pool.save_latest_experience(pickle_path)

    def load_experience(self, experience_path):
        self._pool.load_experience(experience_path)


class MultitaskPercentileReplayPool(object):
    def __init__(self,
                 environment,
                 num_tasks=1,
                 percentile=1.0,
                 task_batch_size=8,
                 threshold_samples=0,
                 *args,
                 **kwargs):
        self._num_tasks = num_tasks
        self._current_task = 0
        self._task_pools = dict([(idx, PercentileReplayPool(
            environment=environment,
            percentile=percentile,
            *args,
            **kwargs
        )) for idx in range(num_tasks)])

        self._task_batch_size = task_batch_size
        self._threshold_samples = threshold_samples

    @property
    def _stage(self):
        return int(self.size >= self._threshold_samples)

    def set_task(self, task_id):
        self._current_task = task_id

    def terminate_episode(self):
        self._task_pools[self._current_task].terminate_episode(stage=self._stage)
    
    @property
    def size(self):
        total_size = 0
        for i in range(self._num_tasks):
            total_size += self._task_pools[i].size
        return total_size

    def add_path(self, path):
        self._task_pools[self._current_task].add_path(path, stage=self._stage)

    def _visited_tasks(self, per_task_batch_size):
        visited = []
        for i in range(self._num_tasks):
            if self._task_pools[i].size > per_task_batch_size:
                visited.append(i)
        return visited

    def random_task_indices(self, task_batch_size, per_task_batch_size):
        visited_tasks = self._visited_tasks(per_task_batch_size)
        return np.random.choice(visited_tasks, task_batch_size)

    def concat_batches(self, batches):
        field_names_flat = self._task_pools[0]._pool.fields_flat.keys()
        batches_flat = [flatten(batch) for batch in batches]
        batch_flat = {
            field_name: np.concatenate([batch_flat[field_name] for batch_flat in batches_flat])
            for field_name in field_names_flat
        }
        batch = unflatten(batch_flat)
        return batch

    def random_batch(self, batch_size):
        per_task_batch_size = int(batch_size / self.task_batch_size)
        task_indices = self.random_task_indices(self.task_batch_size, per_task_batch_size)
        batches = [
            self._task_pools[i].random_batch(per_task_batch_size, stage=self._stage)
            for i in task_indices]
        batch = self.concat_batches(batches)
        return batch

    def random_batch_from_task(self, task_id, batch_size):
        return self._task_pools[task_id].random_batch(batch_size)

    @property
    def task_batch_size(self):
        return self._task_batch_size

    def save_latest_experience(self, pickle_path):
        for idx, pool in self._task_pools.items():
            pool.save_latest_experience(pickle_path.format(task_id=idx))

    def load_experience(self, experience_path):
        for idx, pool in self._task_pools.items():
            pool.load_experience(experience_path.format(task_id=idx))
