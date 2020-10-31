# -*- coding: utf-8 -*-
"""3_multiprocessing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/3_multiprocessing.ipynb

# Stable Baselines Tutorial - Multiprocessing of environments

Github repo: https://github.com/araffin/rl-tutorial-jnrr19

Stable-Baselines: https://github.com/hill-a/stable-baselines

Documentation: https://stable-baselines.readthedocs.io/en/master/

RL Baselines zoo: https://github.com/araffin/rl-baselines-zoo


## Introduction

In this notebook, you will learn how to use *Vectorized Environments* (aka multiprocessing) to make training faster. You will also see that this speed up comes at a cost of sample efficiency.

## Install Dependencies and Stable Baselines Using Pip
"""

# Commented out IPython magic to ensure Python compatibility.
# Stable Baselines only supports tensorflow 1.x for now
# %tensorflow_version 1.x

"""### Remove tensorflow warnings

To have a clean output, we will filter tensorflow warnings, mostly due to the migration from tf 1.x to 2.x
"""

# Filter tensorflow version warnings
import os
# https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints/40426709
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # or any {'0', '1', '2'}
import warnings
# https://stackoverflow.com/questions/15777951/how-to-suppress-pandas-future-warning
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=Warning)
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.autograph.set_verbosity(0)
import logging
tf.get_logger().setLevel(logging.ERROR)

"""## Vectorized Environments and Imports

[Vectorized Environments](https://stable-baselines.readthedocs.io/en/master/guide/vec_envs.html) are a method for stacking multiple independent environments into a single environment. Instead of training an RL agent on 1 environment per step, it allows us to train it on n environments per step. This provides two benefits:
* Agent experience can be collected more quickly
* The experience will contain a more diverse range of states, it usually improves exploration

Stable-Baselines provides two types of Vectorized Environment:
- SubprocVecEnv which run each environment in a separate process
- DummyVecEnv which run all environment on the same process

In practice, DummyVecEnv is usually faster than SubprocVecEnv because of communication delays that subprocesses have.
"""

# Commented out IPython magic to ensure Python compatibility.
import time
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

import gym

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds
from stable_baselines import PPO2

"""Import evaluate function"""

from stable_baselines.common.evaluation import evaluate_policy

"""## Define an environment function

The multiprocessing implementation requires a function that can be called inside the process to instantiate a gym env
"""

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        # Important: use a different seed for each environment
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

"""Stable-Baselines also provides directly an helper to create vectorized environment:"""

from stable_baselines.common.cmd_util import make_vec_env

"""## Define a few constants (feel free to try out other environments and algorithms)
We will be using the Cartpole environment: [https://gym.openai.com/envs/CartPole-v1/](https://gym.openai.com/envs/CartPole-v1/)

![Cartpole](https://cdn-images-1.medium.com/max/1143/1*h4WTQNVIsvMXJTCpXm_TAw.gif)
"""

env_id = 'CartPole-v1'
# The different number of processes that will be used
PROCESSES_TO_TEST = [1, 2, 4, 8, 16] 
NUM_EXPERIMENTS = 3 # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
TRAIN_STEPS = 5000
# Number of episodes for evaluation
EVAL_EPS = 20
ALGO = PPO2

# We will create one environment to evaluate the agent on
eval_env = gym.make(env_id)

"""## Iterate through the different numbers of processes

For each processes, several experiments are run per process
This may take a couple of minutes.
"""

reward_averages = []
reward_std = []
training_times = []
total_procs = 0
for n_procs in PROCESSES_TO_TEST:
    total_procs += n_procs
    print('Running for n_procs = {}'.format(n_procs))
    if n_procs == 1:
        # if there is only one process, there is no need to use multiprocessing
        train_env = DummyVecEnv([lambda: gym.make(env_id)])
    else:
        # Here we use the "spawn" method for launching the processes, more information is available in the doc
        # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))
        train_env = SubprocVecEnv([make_env(env_id, i+total_procs) for i in range(n_procs)], start_method='spawn')

    rewards = []
    times = []

    for experiment in range(NUM_EXPERIMENTS):
        # it is recommended to run several experiments due to variability in results
        train_env.reset()
        model = ALGO('MlpPolicy', train_env, verbose=0)
        start = time.time()
        model.learn(total_timesteps=TRAIN_STEPS)
        times.append(time.time() - start)
        mean_reward, _  = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
        rewards.append(mean_reward)
    # Important: when using subprocess, don't forget to close them
    # otherwise, you may have memory issues when running a lot of experiments
    train_env.close()
    reward_averages.append(np.mean(rewards))
    reward_std.append(np.std(rewards))
    training_times.append(np.mean(times))

"""## Plot the results"""

training_steps_per_second = [TRAIN_STEPS / t for t in training_times]

plt.figure(figsize=(9, 4))
plt.subplots_adjust(wspace=0.5)
plt.subplot(1, 2, 1)
plt.errorbar(PROCESSES_TO_TEST, reward_averages, yerr=reward_std, capsize=2)
plt.xlabel('Processes')
plt.ylabel('Average return')
plt.subplot(1, 2, 2)
plt.bar(range(len(PROCESSES_TO_TEST)), training_steps_per_second)
plt.xticks(range(len(PROCESSES_TO_TEST)), PROCESSES_TO_TEST)
plt.xlabel('Processes')
_ = plt.ylabel('Training steps per second')

"""## Sample efficiency vs wall clock time trade-off
There is clearly a trade-off between sample efficiency, diverse experience and wall clock time. Lets try getting the best performance in a fixed amount of time, say 10 seconds per experiment
"""

SECONDS_PER_EXPERIMENT = 10
steps_per_experiment = [int(SECONDS_PER_EXPERIMENT * fps) for fps in training_steps_per_second]
reward_averages = []
reward_std = []
training_times = []

for n_procs, train_steps in zip(PROCESSES_TO_TEST, steps_per_experiment):
    total_procs += n_procs
    print('Running for n_procs = {} for steps = {}'.format(n_procs, train_steps))
    if n_procs == 1:
        # if there is only one process, there is no need to use multiprocessing
        train_env = DummyVecEnv([lambda: gym.make(env_id)])
    else:
        train_env = SubprocVecEnv([make_env(env_id, i+total_procs) for i in range(n_procs)], start_method='spawn')
        # Alternatively, you can use a DummyVecEnv if the communication delays is the bottleneck
        # train_env = DummyVecEnv([make_env(env_id, i+total_procs) for i in range(n_procs)])

    rewards = []
    times = []

    for experiment in range(NUM_EXPERIMENTS):
        # it is recommended to run several experiments due to variability in results
        train_env.reset()
        model = ALGO('MlpPolicy', train_env, verbose=0)
        start = time.time()
        model.learn(total_timesteps=train_steps)
        times.append(time.time() - start)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
        rewards.append(mean_reward)

    train_env.close()
    reward_averages.append(np.mean(rewards))
    reward_std.append(np.std(rewards))
    training_times.append(np.mean(times))

"""## Plot the results"""

training_steps_per_second = [s / t for s,t in zip(steps_per_experiment, training_times)]

plt.figure()
plt.subplot(1,2,1)
plt.errorbar(PROCESSES_TO_TEST, reward_averages, yerr=reward_std, capsize=2, c='k', marker='o')
plt.xlabel('Processes')
plt.ylabel('Average return')
plt.subplot(1,2,2)
plt.bar(range(len(PROCESSES_TO_TEST)), training_steps_per_second)
plt.xticks(range(len(PROCESSES_TO_TEST)),PROCESSES_TO_TEST)
plt.xlabel('Processes')
plt.ylabel('Training steps per second')

"""## DummyVecEnv vs SubprocVecEnv"""

reward_averages = []
reward_std = []
training_times = []
total_procs = 0
for n_procs in PROCESSES_TO_TEST:
    total_procs += n_procs
    print('Running for n_procs = {}'.format(n_procs))
    # Here we are using only one process even for n_env > 1
    # this is equivalent to DummyVecEnv([make_env(env_id, i + total_procs) for i in range(n_procs)])
    train_env = make_vec_env(env_id, n_envs=n_procs)

    rewards = []
    times = []

    for experiment in range(NUM_EXPERIMENTS):
        # it is recommended to run several experiments due to variability in results
        train_env.reset()
        model = ALGO('MlpPolicy', train_env, verbose=0)
        start = time.time()
        model.learn(total_timesteps=TRAIN_STEPS)
        times.append(time.time() - start)
        mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
        rewards.append(mean_reward)

    train_env.close()
    reward_averages.append(np.mean(rewards))
    reward_std.append(np.std(rewards))
    training_times.append(np.mean(times))

training_steps_per_second = [TRAIN_STEPS / t for t in training_times]

plt.figure()
plt.subplot(1,2,1)
plt.errorbar(PROCESSES_TO_TEST, reward_averages, yerr=reward_std, capsize=2)
plt.xlabel('Processes')
plt.ylabel('Average return')
plt.subplot(1,2,2)
plt.bar(range(len(PROCESSES_TO_TEST)), training_steps_per_second)
plt.xticks(range(len(PROCESSES_TO_TEST)),PROCESSES_TO_TEST)
plt.xlabel('Processes')
plt.ylabel('Training steps per second')

"""### What's happening?

It seems that having only one process for n environments is faster in our case.
In practice, the bottleneck does not come from the environment computation, but from synchronisation and communication between processes. To learn more about that problem, you can start [here](https://github.com/hill-a/stable-baselines/issues/322#issuecomment-492202915)

## Conclusions
This notebook has highlighted some of the pros and cons of multiprocessing. It is worth mentioning that colab notebooks only provide two CPU cores per process, so we do not see a linear scaling of the FPS of the environments. State of the art Deep RL research has scaled parallel processing to tens of thousands of CPU cores, [OpenAI RAPID](https://openai.com/blog/how-to-train-your-openai-five/) [IMPALA](https://arxiv.org/abs/1802.01561).

Do you think this direction of research is transferable to real world robots / intelligent agents?

Things to try:
* Another algorithm / environment.
* Increase the number of experiments.
* Train for more iterations.
"""