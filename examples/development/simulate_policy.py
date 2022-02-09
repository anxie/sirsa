import sys
import argparse
from distutils.util import strtobool
import json
import os
import pickle

import tensorflow as tf

from softlearning.environments.utils import get_environment_from_params
from softlearning.policies.utils import get_policy_from_variant
from softlearning.samplers import rollouts, simple_sampler
from softlearning.replay_pools.utils import get_replay_pool_from_variant
from softlearning.value_functions.utils import get_Q_function_from_variant
from softlearning.algorithms.utils import get_algorithm_from_variant

import numpy as np
import matplotlib.pyplot as plt

import multiprocessing
from natsort import natsorted


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_path',
                        type=str,
                        help='Path to the checkpoint.')
    parser.add_argument('--max-path-length', '-l', type=int, default=500)
    parser.add_argument('--num-rollouts', '-n', type=int, default=1)
    parser.add_argument('--env-type', '-t', type=str, default='uniform')
    parser.add_argument('--task-ids', '-i', nargs='+', type=int, default=[])
    parser.add_argument('--sample-ids', '-s', nargs='+', type=int, default=[])
    parser.add_argument('--deterministic', '-d',
                        type=lambda x: bool(strtobool(x)),
                        nargs='?',
                        const=True,
                        default=True,
                        help="Evaluate policy deterministically.")

    args = parser.parse_args(argv)

    return args


def simulate_policy(args):
    experiment_path = args.experiment_path.rstrip('/')

    trial_paths = []
    for d in os.listdir(experiment_path):
        if '=' in d:
            trial_path = os.path.join(experiment_path, d)
            ckpt = natsorted([c for c in os.listdir(trial_path) if 'checkpoint_' in c])[-1]
            trial_paths.append(trial_path)

    task_ids = np.arange(args.task_ids[0], args.task_ids[-1])

    if args.sample_ids:
        sample_ids = args.sample_ids
    elif args.env_type == 'uniform' or args.env_type == 'gaussian':
        sample_ids = np.arange(50)
    else:
        sample_ids = [-1]

    manager = multiprocessing.Manager()
    return_dict = manager.dict()
    jobs = []
    for i, trial_path in enumerate(trial_paths):
        worker_args = (i, args, trial_path, ckpt, task_ids, sample_ids, return_dict)
        p = multiprocessing.Process(target=worker, args=worker_args)
        jobs.append(p)
        p.start()

    for p in jobs:
        p.join()

    returns = np.array(return_dict.values())
    returns = returns.reshape(-1, len(task_ids), len(sample_ids)).transpose((1,2,0))    # (# tasks, # samples, # trials)

    means = returns.mean(-1)
    stds = returns.std(-1)

    min_args = np.argsort(means, 1)[:, 0]
    scores = returns[np.arange(20), min_args, :].mean(0)

    mean = scores.mean()
    std = scores.std()

    print(mean, std / np.sqrt(len(trial_paths)))


def worker(procnum,
           args,
           trial_path,
           checkpoint,
           task_ids,
           sample_ids,
           return_dict):
    print('Process {}'.format(procnum))

    session = tf.keras.backend.get_session()

    variant_path = os.path.join(trial_path, 'params.pkl')
    with open(variant_path, 'rb') as f:
        variant = pickle.load(f)

    checkpoint_path = os.path.join(trial_path, checkpoint)
    with session.as_default():
        pickle_path = os.path.join(checkpoint_path, 'checkpoint.pkl')
        with open(pickle_path, 'rb') as f:
            picklable = pickle.load(f)

    environment_params = (
        variant['environment_params']['evaluation']
        if 'evaluation' in variant['environment_params']
        else variant['environment_params']['training'])

    environment_params['kwargs'].update({'env_type': args.env_type})
    evaluation_environment = get_environment_from_params(environment_params)

    variant['algorithm_params']['kwargs']['predict_context'] = False

    policy = (
        get_policy_from_variant(variant, evaluation_environment))
    policy.set_weights(picklable['policy_weights'])

    if variant['algorithm_params']['kwargs'].get('latent_dim', 0):
        Qs = get_Q_function_from_variant(
            variant, evaluation_environment)

        replay_pool = (
            get_replay_pool_from_variant(variant, evaluation_environment))

        algorithm = get_algorithm_from_variant(
            variant=variant,
            training_environment=evaluation_environment,
            evaluation_environment=evaluation_environment,
            policy=policy,
            initial_exploration_policy=policy,
            Qs=Qs,
            pool=replay_pool,
            sampler=None,
            session=session)
        if algorithm._ensemble_size > 1:
            for i in range(algorithm._ensemble_size):
                algorithm._encoders[i].set_weights(picklable['encoder_weights'][i])
        else:
            algorithm._encoders[0].set_weights(picklable['encoder_weights'])
        if variant['algorithm_params']['kwargs']['predict_context']:
            algorithm._predictor.set_weights(picklable['predictor_weights'])
    else:
        algorithm = None

    returns = []
    for task_id in task_ids:
        for sample_id in sample_ids:
            print('Task id: {}, Sample id: {}'.format(task_id, sample_id))
            with policy.set_deterministic(args.deterministic):
                paths = rollouts(args.num_rollouts,
                                 evaluation_environment,
                                 policy,
                                 alg=algorithm,
                                 latent_dim=variant['algorithm_params']['kwargs'].get('latent_dim', 0),
                                 update_belief=variant['sampler_params']['kwargs'].get('update_belief', False),
                                 path_length=args.max_path_length,
                                 task_id=task_id,
                                 sample_id=sample_id)

            r = [np.sum(path['rewards']) for path in paths]
            returns.append(r)

    returns = np.array(returns).reshape(len(task_ids), len(sample_ids), args.num_rollouts).mean(-1)
    return_dict[procnum] = returns


def main(argv=None):
    args = parse_args(argv)
    simulate_policy(args)


if __name__ == '__main__':
    main(argv=sys.argv[1:])
