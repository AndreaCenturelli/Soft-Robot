from stable_baselines.common.env_checker import check_env
import tensorflow as tf
import gym
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


env_dict = gym.envs.registration.registry.env_specs.copy()
for env in env_dict:
    if 'soft_robot-v0' in env:
        print("Remove {} from registry".format(env))
        del gym.envs.registration.registry.env_specs[env]
# import foo_env


import os
os.environ['OPENAI_LOGDIR'] = '/tmp'
os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'
    
import soft_robot_env
env=gym.make('soft_robot-v0')

import time
# start=time.time()
from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines import PPO1,PPO2
from stable_baselines.common.policies import MlpPolicy
# model=PPO1(MlpPolicy,env,verbose=1,optim_batchsize=64,timesteps_per_actorbatch=256,
#            n_cpu_tf_sess=11,full_tensorboard_log=True,
#             tensorboard_log='/tmp',clip_param=0.3,)
# model=PPO2(MlpPolicy,env,n_steps=512,n_cpu_tf_sess=11,verbose=1,tensorboard_log='/tmp',
#            full_tensorboard_log=True,)
# model.learn(total_timesteps=10000,log_interval=10,reset_num_timesteps=False)

# end=time.time()-start
# print(end)


from stable_baselines.common import set_global_seeds

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



env_id='soft_robot-v0'

PROCESSES_TO_TEST = [1,4,12] 
NUM_EXPERIMENTS = 1 # RL algorithms can often be unstable, so we run several experiments (see https://arxiv.org/abs/1709.06560)
TRAIN_STEPS = 200
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

from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
import numpy as np

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
        print("using dummy")
    else:
        # Here we use the "spawn" method for launching the processes, more information is available in the doc
        # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='spawn'))
        train_env = SubprocVecEnv([make_env(env_id, i+total_procs) for i in range(n_procs)])
        print("using subvecenv")
    rewards = []
    times = []

    
    # it is recommended to run several experiments due to variability in results
    train_env.reset()
    model = PPO2('MlpPolicy', train_env, verbose=0,n_cpu_tf_sess=1,tensorboard_log='/tmp',
            full_tensorboard_log=True)
    start = time.time()
    model.learn(total_timesteps=TRAIN_STEPS)
    times.append(time.time() - start)
    print("\n \n \n done")
    # mean_reward, _  = evaluate_policy(model, eval_env, n_eval_episodes=EVAL_EPS)
    # rewards.append(mean_reward)
    # Important: when using subprocess, don't forget to close them
    # otherwise, you may have memory issues when running a lot of experiments
    train_env.close()
    # reward_averages.append(np.mean(rewards))
    # reward_std.append(np.std(rewards))
    training_times.append(np.mean(times))

















