from stable_baselines.common.env_checker import check_env
import tensorflow as tf
import gym
from soft_robot_env.envs.soft_env_functions import EnvironmentElastica
import numpy as np
import time 
from gym import spaces

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


class SoftRobotEnvironment(gym.Env):
    metadata = {'render.modes': ['console']}
    def __init__(self,random_target=False):
        super(SoftRobotEnvironment, self).__init__()
        self.random_target=random_target
        numeric_low=-np.ones((18,),dtype=np.float64)
        numeric_high=np.ones((18,),dtype=np.float64)
        self.observation_space = spaces.Box(low=numeric_low, high=numeric_high,
                                            dtype=np.float64)
        numeric_low=-8*np.ones((15,),dtype=np.float16)
        numeric_high=8*np.ones((15,),dtype=np.float16)
        self.action_space=spaces.Box(low=numeric_low, high=numeric_high, 
                                     dtype=np.float16)
        self.reward_range=(-np.inf,110)
        
        self.el_env=EnvironmentElastica(dt=2.5e-5,sim_time=0.01,
                                        random_target=self.random_target)
        
        self.step_num=1
        self.viewer=None
        self.state=None
        
    def step(self,action):
        print("\n Step number: ",self.step_num,"\n")
        self.step_num+=1
        obs,reward,done,info=self.el_env.step(action)
        # if self.step_num==40:
        #     done=True
        self.state=obs
        return obs.flatten(),reward,done,info
    
    def reset(self):
        del self.el_env
        self.el_env=EnvironmentElastica(dt=2.5e-5,sim_time=0.01,
                                        random_target=self.random_target)
        return self.el_env.in_pos.flatten()
        
    def render(self,mode='human'):
        screen_width = 400
        screen_height = 220
        scale=200
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.end_effector=rendering.make_circle(radius=10)
            self.end_effector.set_color(1,0,0)
            self.ee_trans=rendering.Transform()
            self.end_effector.add_attr(self.ee_trans)
            top=screen_height
            mid=scale
            arm_base=rendering.make_polygon([(mid-5,top-5),(mid-5,top),
                                                  (mid+5,top),(mid+5,top-5)])
            arm_base.set_color(0,0,1)
            self.tar=rendering.make_circle(radius=10)
            self.tar.set_color(0,1,0)
            self.tar_trans=rendering.Transform()
            self.tar.add_attr(self.tar_trans)
            
            self.viewer.add_geom(arm_base)
            self.viewer.add_geom(self.tar)
            self.viewer.add_geom(self.end_effector)
        
        if self.state is None:
            return None
        x=self.state
        ee_x=x[0,-2]*scale +screen_width/2
        ee_y=screen_height+x[1,-2]*scale
        ee_z=screen_height+x[2,-2]*scale
        tar_x=x[0,-1]*scale +screen_width/2
        tar_y=x[1,-1]*scale
        tar_z=screen_height+x[2,1]*scale
    
        self.ee_trans.set_translation(ee_x, ee_z)
        self.tar_trans.set_translation(tar_x, tar_z)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

# # from gym import wrappers

    

# # env1 = gym.make('CartPole-v0')
# # # env1 = wrappers.Monitor(env1, '/tmp/cartpole-experiment-1', force=True)

# # action=env1.action_space.sample()
# # observation=env1.reset()
# # observation, reward, done, info = env1.step(action)
# # env1.render()
# import os
# os.environ['OPENAI_LOGDIR'] = '/tmp'
# os.environ['OPENAI_LOG_FORMAT'] = 'stdout,tensorboard'
    
# env=gym.make('soft_robot-v0')
# # env.action_space
# done=False
# while not done:
#     forze=env.action_space.sample()
#     obs,reward,done,info=env.step(forze)
#     env.render()
# env.close()
    
    
# from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines import PPO1
# from stable_baselines.common.policies import MlpPolicy
# model=PPO1(MlpPolicy,env,verbose=1,optim_batchsize=64,n_cpu_tf_sess=11,full_tensorboard_log=True,
#            tensorboard_log='/tmp')
# model.learn(total_timesteps=1000,log_interval=10)


# from stable_baselines.common.evaluation import evaluate_policy
# from stable_baselines import PPO1
# from stable_baselines.common.policies import MlpPolicy
# from gym import 
# env1 = 
# model1=PPO1(MlpPolicy,env1,verbose=1,n_cpu_tf_sess=12,full_tensorboard_log=True,
#             tensorboard_log='/tmp')
# model1.learn(1000000,log_interval=10)




