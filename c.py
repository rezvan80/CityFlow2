import cityflow
print(cityflow.__version__)

eng = cityflow.Engine("/content/CityFlow2/examples/config.json", thread_num=1)
print("CityFlow installed successfully!")

import numpy as np
cell_nums=12

cell_lengths=np.array([0,49,109,149,199,220,234,240,250,255,260,265,269])
cell_lengths=np.array([0,49,109,149,199,220,234,240,250,255,260,265,269])

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
cell_nums=12

cell_lengths=np.array([0,4,14,35,70,120,160,220,269])
cell_lengths=np.array([0,4,10,14,20,25,35,48,70,120,160,220,269])
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


reward=[]
class CityFlowEnv(gym.Env):
    def __init__(self):
        super(CityFlowEnv, self).__init__()
        self.action_space = Discrete(8)
        self.observation_space = Box(low=0.0,high=1.0,shape=(8,12), dtype=np.float32)

    def step(self, action):
        eng.set_tl_phase("intersection_1_1",action )
        eng.next_step()
        road_0_1_0_celled_vehicle_nums=cell("road_0_1_0")
        road_1_0_1_celled_vehicle_nums=cell("road_1_0_1")
        road_2_1_2_celled_vehicle_nums=cell("road_2_1_2")
        road_1_2_3_celled_vehicle_nums=cell("road_1_2_3")
        road_1_1_0_celled_vehicle_nums=cell2("road_1_1_0")
        road_1_1_1_celled_vehicle_nums=cell2("road_1_1_1")
        road_1_1_3_celled_vehicle_nums=cell2("road_1_1_3")
        road_1_1_2_celled_vehicle_nums=cell2("road_1_1_2")
        observation=np.array([road_0_1_0_celled_vehicle_nums,
              road_1_0_1_celled_vehicle_nums,
              road_2_1_2_celled_vehicle_nums,
              road_1_2_3_celled_vehicle_nums,
              road_1_1_2_celled_vehicle_nums,
              road_1_1_3_celled_vehicle_nums,
              road_1_1_0_celled_vehicle_nums,
              road_1_1_1_celled_vehicle_nums])
        self.reward=1/sum(eng.get_lane_vehicle_count().values())
        reward.append(self.reward)
        avg_reward=sum(reward)/len(reward)


        if eng.get_current_time()>1000:

          reward.clear()
          print(avg_reward)
          self.done=True

        info={}

        return observation, self.reward, self.done, info

    def reset(self ):
        self.done=False
        eng.reset()
        eng.next_step()
        road_0_1_0_celled_vehicle_nums=cell("road_0_1_0")
        road_1_0_1_celled_vehicle_nums=cell("road_1_0_1")
        road_2_1_2_celled_vehicle_nums=cell("road_2_1_2")
        road_1_2_3_celled_vehicle_nums=cell("road_1_2_3")
        road_1_1_0_celled_vehicle_nums=cell2("road_1_1_0")
        road_1_1_1_celled_vehicle_nums=cell2("road_1_1_1")
        road_1_1_3_celled_vehicle_nums=cell2("road_1_1_3")
        road_1_1_2_celled_vehicle_nums=cell2("road_1_1_2")
        observation=np.array([road_0_1_0_celled_vehicle_nums,
              road_1_0_1_celled_vehicle_nums,
              road_2_1_2_celled_vehicle_nums,
              road_1_2_3_celled_vehicle_nums,
              road_1_1_2_celled_vehicle_nums,
              road_1_1_3_celled_vehicle_nums,
              road_1_1_0_celled_vehicle_nums,
              road_1_1_1_celled_vehicle_nums])

        return observation



from binascii import a2b_base64
import torch as th
import torch.nn as nn
import math

import torch
import torch.nn as nn
from torch.optim import SGD

import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from typing import Tuple
# Custom Self-Attention Network
class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.norm1 = nn.LayerNorm(12)
        self.norm2 = nn.LayerNorm(12)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.self_attn = nn.MultiheadAttention(embed_dim=12, num_heads=1)
        self.model=nn.Sequential(nn.Linear(12,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,12),nn.Tanh())




    def forward(self, x: th.Tensor) -> th.Tensor:

        attn_output, _ = self.self_attn(x, x, x)
        x = x + attn_output
        x = self.norm1(x)

        ff_output= self.model(x)
        x = x + ff_output
        x = self.norm2(x)
        return x

class SelfAttention(nn.Module):
    def __init__(self, d, d_q, d_k, d_v):
        super(SelfAttention, self).__init__()
        self.latent_dim_pi = 8
        self.latent_dim_vf = 8
        self.model1=nn.Sequential(nn.Linear(96,8),nn.Tanh())
        self.model2=nn.Sequential(nn.Linear(96,8),nn.Tanh())
        self.encoderlayer1=EncoderLayer()
        self.encoderlayer2=EncoderLayer()
        self.encoderlayer3=EncoderLayer()
      #  self.encoderlayer4=EncoderLayer()
     #   self.encoderlayer5=EncoderLayer()
       # self.encoderlayer6=EncoderLayer()
        self.encoderlayer7=EncoderLayer()
        self.encoderlayer8=EncoderLayer()
        self.encoderlayer9=EncoderLayer()
       # self.encoderlayer10=EncoderLayer()
      #  self.encoderlayer11=EncoderLayer()
      #  self.encoderlayer12=EncoderLayer()
    def forward(self, x: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:


      #context_vector2=self.policy_net2(context_vector2)
      return self.forward_actor(x),self.forward_critic(x)


    def forward_actor(self, x: th.Tensor) -> th.Tensor:
        x=self.encoderlayer1(x)
        x=self.encoderlayer2(x)
        x=self.encoderlayer3(x)

        x=self.model1(x.reshape(-1,96))
        return x

    def forward_critic(self, x: th.Tensor) -> th.Tensor:

        x=self.encoderlayer7(x)
        x=self.encoderlayer8(x)
        x=self.encoderlayer9(x)
        x=self.model2(x.reshape(-1,96))
      #  doc.add_paragraph(f"In-Projection Weights:\n{self.encoderlayer7.self_attn.in_proj_weight.data}")
       # doc.add_paragraph(f"Out-Projection Weights:\n{self.encoderlayer7.self_attn.out_proj.weight.data}")


       # print("Weights have been saved to weights_log.docx")
        return x



# Custom Policy incorporating the Self-Attention feature extractor
class CustomPolicy(ActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule, *args, **kwargs):

        use_sde = kwargs.pop('use_sde', False)
        super(CustomPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,


            *args,
            use_sde=use_sde,
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:

        # Use the shared extracted features for both policy and value networks
        self.features_extractor = nn.Identity()
        self.mlp_extractor = SelfAttention(8,8,8,8)
        self.pi_features_extractor= nn.Identity()
        self.vf_features_extractor=nn.Identity()
        self.action_net=nn.Identity()
        self.value_net=nn.Identity()
        
from stable_baselines3 import PPO
import os

import time


from stable_baselines3 import PPO
import os

import time



# Define directories within the Colab's working directory
models_dir = "/content/models"
logdir = "/content/logs"

# Ensure these directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(logdir, exist_ok=True)

# Append timestamp for current session directories
session_model_dir = f"{models_dir}/{int(time.time())}"
session_log_dir = f"{logdir}/{int(time.time())}"

# Make sure session directories are created
os.makedirs(session_model_dir, exist_ok=True)
os.makedirs(session_log_dir, exist_ok=True)

print(f"Model directory: {session_model_dir}")
print(f"Log directory: {session_log_dir}")

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = CityFlowEnv()
env.reset()

model = PPO(CustomPolicy, env, verbose=1, tensorboard_log=session_log_dir)

TIMESTEPS = 200000

# Sum the number of elements in all parameters (weights) of the model
total_params = sum(p.numel() for p in model.policy.parameters())

print(f"Total number of weights in the model: {total_params}")

model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False, tb_log_name=f"PPO")
model.save(session_model_dir)
