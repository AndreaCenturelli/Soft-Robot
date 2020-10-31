import numpy as np

# FIXME without appending sys.path make it more generic
import sys

# sys.path.append("../../")

import os
from collections import defaultdict
from elastica.wrappers import BaseSystemCollection, Constraints, Forcing, CallBacks
from elastica.rod.cosserat_rod import CosseratRod
from elastica.external_forces import SoftRobotForces
from elastica.boundary_conditions import OneEndFixedRod, FreeRod
from elastica.callback_functions import CallBackBaseClass
from elastica.timestepper.symplectic_steppers import PositionVerlet, PEFRL
from elastica.timestepper import integrate
from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d
import random
import time


class FlagellaSimulator(BaseSystemCollection, Constraints, Forcing, CallBacks):
    pass

class EnvironmentElastica(object):
    def __init__(self,dt=2.5e-5,sim_time=0.1,random_target=True): #### put params in txt or whatev
        self.shearable_rod=self.rod()
        self.in_pos=self.shearable_rod.position_collection.copy()
        self.ee_in_pos=self.in_pos[:,-1]
        self.render_array={}
        self.render_array['arm_position']=self.in_pos
        self.random_target=random_target

        self.target=self.create_target(self.random_target)
        self.in_pos=np.column_stack((self.in_pos[:,[10,20,30,40,50]],self.target))

        self.render_array['target_position']=self.target.copy()
        self.dt=dt
        self.sim_time=sim_time
        
        self.prev_dist=np.sqrt(np.sum((self.ee_in_pos - self.target) ** 2))
        self.target_radius=0.1
        self.reward_bias=-75/(self.target_radius-self.prev_dist)*self.target_radius+75
        self.reward_slope=75/(self.target_radius-self.prev_dist)
        
        
    def step(self,end_force):
        position,velocity=self._rod_step(end_force)
        ee_pos=position[:,-1]
        self._target_step()
        reward=self._get_reward(ee_pos)
        obs=np.column_stack((position,self.target))
        if reward==110:
            done=True
        else:
            done=False
        info=self.render_array
        return obs,reward,done,info
        
        
        
    def rod(self):
        n_elem = 50
        start = np.zeros((3,))
        direction = np.array([0.0, 0.0, -1.0])
        normal = np.array([0.0, 1.0, 0.0])
        base_length = 1.0
        base_radius = 0.025
        base_area = np.pi * base_radius ** 2
        density = 1000
        nu = 5.0
        E = 1e7
        poisson_ratio = 0.5
    
        shearable_rod = CosseratRod.straight_rod(
            n_elem,
            start,
            direction,
            normal,
            base_length,
            base_radius,
            density,
            nu,
            E,
            poisson_ratio,
        )
        return shearable_rod
    
    def create_target(self,random_tar):
        if random_tar==True:
            x=1-2*random.random()
            y=1-2*random.random()
            z=-random.random()
        else:
            x=0.5
            y=0.5
            z=-0.5
        return np.array([x,y,z])
    
    def _get_reward(self,ee_pos):
        dist=np.sqrt(np.sum((ee_pos -self.target) ** 2))
        print("\n The distance is: ",dist,"\n")
        reward=self.reward_slope*dist+self.reward_bias
        if reward==75: #additional reward for touching the ball
            reward+=25
        # if dist<self.prev_dist-0.05: #additional reward for being better than before
        #     reward+=10
        # elif dist>self.prev_dist+0.05: #punishment for being worse than before
        #     reward-=11
        self.prev_dist=dist
        return reward
    def _target_step(self):
        pass
    
    def _rod_step(self,end_force):
        flagella_sim = FlagellaSimulator()
        flagella_sim.append(self.shearable_rod)
        #******************** FIX ONE END **************+++
        flagella_sim.constrain(self.shearable_rod).using(
        OneEndFixedRod, 
        constrained_position_idx=(0,), 
        constrained_director_idx=(0,)
    )
        
        ramp_up_time =1e-4
        flagella_sim.add_forcing_to(self.shearable_rod).using(
            SoftRobotForces, 
            end_force, 
            ramp_up_time=ramp_up_time
        )
    
        flagella_sim.finalize()
        
        timestepper = PositionVerlet()
    
        final_time = (self.sim_time+ self.dt) 
        total_steps = int(final_time / self.dt)
        integrate(timestepper, flagella_sim, final_time, total_steps)
        position=self.shearable_rod.position_collection.copy()
        velocity=self.shearable_rod.velocity_collection.copy()
        self.render_array["arm_position"]=position
        
        return position[:,[10,20,30,40,50]],velocity
    # def plot_positions_3d(self):
        
    #     fig = plt.figure(figsize=(12,12)) #Adjusts the aspect ratio and enlarges the figure (text does not enlarge)
    #     ax = fig.gca(projection='3d')
    #     ax.set_xlabel('x')
    #     ax.set_ylabel('y')
    #     # ax.view_init(elev=45, azim=45)
    #     ax.set_zlabel('z')
    #     ax.set_xlim3d(-1, 1)
    #     ax.set_ylim3d(-1,1)
    #     ax.set_zlim3d(-1,0)
    #     for col in range(self.position.shape[1]):
    #         print(col)
    #         x=self.position[0][col]
    #         y=self.position[1][col]
    #         z=self.position[2][col]
    #         ax.plot(x, y,z, "k") 
    #     for col in [10,20,30,40]:
    #         x=float(self.position[0][col])
    #         y=float(self.position[1][col])
    #         z=float(self.position[2][col])
    #         ax.scatter(x, y,z, c="b",marker=".") 
    #     x=float(self.position[0][-1])
    #     y=float(self.position[1][-1])
    #     z=float(self.position[2][-1])
    #     ax.scatter(x, y,z, c="g",marker="o")  
    #     x=float(self.position[0][0])
    #     y=float(self.position[1][0])
    #     z=float(self.position[2][0])
    #     ax.scatter(x, y,z, c="g",marker="^")
    def render_to_binary(self):
        pass
       
       
# if __name__ == "__main__":
    
#     forze=np.zeros((5,3))
    
#     env=EnvironmentElastica(dt=2.5e-5,sim_time=0.1,random_target=False)
#     start=time.time()
#     obs1,reward,done,info = env.step(forze)
#     # print("Final real time: ",time.time()-start)
#     print("Reward:  ",reward)
#     print("Observation:  ",obs1)
    
#     forze=np.array([[-1,0,0],[-2,0,0],[-3,0,0],[-4,0,0],[-5,0,0]])
#     start=time.time()
#     obs2,reward,done,info = env.step(forze)
#     # print("Final real time: ",time.time()-start)    
#     print("Reward:  ",reward)
#     print("Observation:  ",obs2)

