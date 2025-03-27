import cityflow
print(cityflow.__version__)

eng = cityflow.Engine("/content/CityFlow2/examples/config.json", thread_num=1)
print("CityFlow installed successfully!")

import numpy as np
cell_nums=8
import gymnasium as gym
cell_lengths=np.array([0,49,109,149,199,234,255,265,269])

cell_nums=8

cell_lengths=np.array([0,49,109,149,199,234,255,265,269])

def cell (road_id):
   cell_1=np.zeros(cell_nums)
   for vehicle in eng.get_vehicles():
        drivable=eng.get_vehicle_info(vehicle).get('drivable')
        distance=float(eng.get_vehicle_info(vehicle).get('distance'))
        if "TO" not in drivable:
             for cell_num in range(cell_nums):
               if road_id in drivable and cell_lengths[cell_num]<distance<cell_lengths[cell_num+1] :
                  cell_1[cell_num]+=1


   return(cell_1)

print(cell("road_0_1_0"))


import numpy as np
cell_nums=8

cell_lengths=np.array([0,4,14,35,70,120,160,220,269])
def cell2 (road_id):
   cell_1=np.zeros(cell_nums)
   for vehicle in eng.get_vehicles():
        drivable=eng.get_vehicle_info(vehicle).get('drivable')
        distance=float(eng.get_vehicle_info(vehicle).get('distance'))
        if "TO" not in drivable:
             for cell_num in range(cell_nums):
               if road_id in drivable and cell_lengths[cell_num]<distance<cell_lengths[cell_num+1] :
                  cell_1[cell_nums-cell_num-1]+=1


   return(cell_1)

print(cell("road_2_1_2"))

import  gym
from gym.spaces import Discrete
from gym.spaces import Box
import numpy as np

import random
import time
from collections import deque

import  gym
from gym.spaces import Discrete
from gym.spaces import Box
import numpy as np

import random
import time
from collections import deque


reward=[]
observations2=[]
observations3=[]
class CityFlowEnv(gym.Env):
    def __init__(self):
        super(CityFlowEnv, self).__init__()
        self.action_space = Discrete(8)
        self.observation_space = Box(low=0.0,high=1.0,shape=(8,8), dtype=np.float32)

    def step(self, action):
        eng.set_tl_phase("intersection_1_1",action )
        eng.next_step()
        road_0_1_0_celled_vehicle_nums=cell("road_0_1_0")
        road_1_0_1_celled_vehicle_nums=cell("road_1_0_1")
        road_2_1_2_celled_vehicle_nums=cell("road_2_1_2")
        road_1_2_3_celled_vehicle_nums=cell("road_1_2_3")
        road_1_1_0_celled_vehicle_nums=cell("road_1_1_0")
        road_1_1_1_celled_vehicle_nums=cell("road_1_1_1")
        road_1_1_3_celled_vehicle_nums=cell("road_1_1_3")
        road_1_1_2_celled_vehicle_nums=cell("road_1_1_2")
        observation=np.array([cell("road_0_1_0_0"),
              cell("road_1_0_1_0"),
              cell("road_2_1_2_0"),
              cell("road_1_2_3_0"),
              cell2("road_1_1_0_0"),
              cell2("road_1_1_1_0"),
              cell2("road_1_1_3_0"),
              cell2("road_1_1_2_0"),
              cell("road_0_1_0_1"),
              cell("road_1_0_1_1"),
              cell("road_2_1_2_1"),
              cell("road_1_2_3_1"),
              cell2("road_1_1_0_1"),
              cell2("road_1_1_1_1"),
              cell2("road_1_1_3_1"),
              cell2("road_1_1_2_1"),
              cell("road_0_1_0_2"),
              cell("road_1_0_1_2"),
              cell("road_2_1_2_2"),
              cell("road_1_2_3_2"),
              cell2("road_1_1_0_2"),
              cell2("road_1_1_1_2"),
              cell2("road_1_1_3_2"),
              cell2("road_1_1_2_2"),
            cell("road_0_1_0_0_3"),
              cell("road_1_0_1_3"),
              cell("road_2_1_2_3"),
              cell("road_1_2_3_3"),
              cell2("road_1_1_0_3"),
              cell2("road_1_1_1_3"),
              cell2("road_1_1_3_3"),
              cell2("road_1_1_2_3"),
           cell("road_0_1_0_4"),
              cell("road_1_0_1_4"),
              cell("road_2_1_2_4"),
              cell("road_1_2_3_4"),
              cell2("road_1_1_0_4"),
              cell2("road_1_1_1_4"),
              cell2("road_1_1_3_4"),
              cell2("road_1_1_2_4"),
         cell("road_0_1_0_5"),
              cell("road_1_0_1_5"),
              cell("road_2_1_2_5"),
              cell("road_1_2_3_5"),
              cell2("road_1_1_0_5"),
              cell2("road_1_1_1_5"),
              cell2("road_1_1_3_5"),
              cell2("road_1_1_2_5"),
          cell("road_0_1_0_6"),
              cell("road_1_0_1_6"),
              cell("road_2_1_2_6"),
              cell("road_1_2_3_6"),
              cell2("road_1_1_0_6"),
              cell2("road_1_1_1_6"),
              cell2("road_1_1_3_6"),
              cell2("road_1_1_2_6"),
          cell("road_0_1_0_7"),
              cell("road_1_0_1_7"),
              cell("road_2_1_2_7"),
              cell("road_1_2_3_7"),
              cell2("road_1_1_0_7"),
              cell2("road_1_1_1_7"),
              cell2("road_1_1_3_7"),
              cell2("road_1_1_2_7")])
        self.done=True
        self.reward=-sum(eng.get_lane_waiting_vehicle_count().values())
        reward.append(self.reward)
        avg_reward=sum(reward)/len(reward)


        if eng.get_current_time()>1000:
          reward.clear()
          print(avg_reward)
          self.done=True

        info={}
        observations2.append(observation.tolist())
        observations3.append([cell("road_0_1_0"),
              cell("road_1_0_1"),
              cell("road_2_1_2"),
              cell("road_1_2_3"),
              cell2("road_1_1_0"),
              cell2("road_1_1_1"),
              cell2("road_1_1_3"),
              cell2("road_1_1_2")])
        observation=np.array([cell("road_0_1_0"),
              cell("road_1_0_1"),
              cell("road_2_1_2"),
              cell("road_1_2_3"),
              cell2("road_1_1_0"),
              cell2("road_1_1_1"),
              cell2("road_1_1_3"),
              cell2("road_1_1_2")])
        return observation, self.reward, self.done, info

    def reset(self ):
        self.done=False
        eng.reset()
        eng.next_step()
        observation=np.array([cell("road_0_1_0_0"),
              cell("road_1_0_1_0"),
              cell("road_2_1_2_0"),
              cell("road_1_2_3_0"),
              cell2("road_1_1_0_0"),
              cell2("road_1_1_1_0"),
              cell2("road_1_1_3_0"),
              cell2("road_1_1_2_0"),
              cell("road_0_1_0_1"),
              cell("road_1_0_1_1"),
              cell("road_2_1_2_1"),
              cell("road_1_2_3_1"),
              cell2("road_1_1_0_1"),
              cell2("road_1_1_1_1"),
              cell2("road_1_1_3_1"),
              cell2("road_1_1_2_1"),
              cell("road_0_1_0_2"),
              cell("road_1_0_1_2"),
              cell("road_2_1_2_2"),
              cell("road_1_2_3_2"),
              cell2("road_1_1_0_2"),
              cell2("road_1_1_1_2"),
              cell2("road_1_1_3_2"),
              cell2("road_1_1_2_2"),
            cell("road_0_1_0_0_3"),
              cell("road_1_0_1_3"),
              cell("road_2_1_2_3"),
              cell("road_1_2_3_3"),
              cell2("road_1_1_0_3"),
              cell2("road_1_1_1_3"),
              cell2("road_1_1_3_3"),
              cell2("road_1_1_2_3"),
           cell("road_0_1_0_4"),
              cell("road_1_0_1_4"),
              cell("road_2_1_2_4"),
              cell("road_1_2_3_4"),
              cell2("road_1_1_0_4"),
              cell2("road_1_1_1_4"),
              cell2("road_1_1_3_4"),
              cell2("road_1_1_2_4"),
         cell("road_0_1_0_5"),
              cell("road_1_0_1_5"),
              cell("road_2_1_2_5"),
              cell("road_1_2_3_5"),
              cell2("road_1_1_0_5"),
              cell2("road_1_1_1_5"),
              cell2("road_1_1_3_5"),
              cell2("road_1_1_2_5"),
          cell("road_0_1_0_6"),
              cell("road_1_0_1_6"),
              cell("road_2_1_2_6"),
              cell("road_1_2_3_6"),
              cell2("road_1_1_0_6"),
              cell2("road_1_1_1_6"),
              cell2("road_1_1_3_6"),
              cell2("road_1_1_2_6"),
          cell("road_0_1_0_7"),
              cell("road_1_0_1_7"),
              cell("road_2_1_2_7"),
              cell("road_1_2_3_7"),
              cell2("road_1_1_0_7"),
              cell2("road_1_1_1_7"),
              cell2("road_1_1_3_7"),
              cell2("road_1_1_2_7")])
        observation=np.array([cell("road_0_1_0"),
              cell("road_1_0_1"),
              cell("road_2_1_2"),
              cell("road_1_2_3"),
              cell2("road_1_1_0"),
              cell2("road_1_1_1"),
              cell2("road_1_1_3"),
              cell2("road_1_1_2")])
        return observation
observations=[]
actions=[]
next_observations=[]
data_ = collections.defaultdict(list)
print(CityFlowEnv().step(random.randint(0, 7)))
for n in range(1000):
  observation=CityFlowEnv().reset()
  for i in range (1000):
    observations.append(observation.tolist())
    action=random.randint(0, 7)
    actions.append(action)
    next_observation=CityFlowEnv().step(action)[0]
    next_observations.append(next_observation.tolist())
    observation=next_observation
print(observations)
print(actions)
print(next_observations)