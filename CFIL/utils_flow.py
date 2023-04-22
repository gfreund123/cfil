#specify

import numpy as np
import torch
import torch.nn as nn
import math
from torch.distributions.normal import Normal
import torch.nn.functional as F
import gym
import os
import json

from utils_data import *

import flow_code.made as made #kamen flows

import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval

import warnings

class Flow(nn.Module):
    def __init__(self, flow=None, flow_type=None, flow_args=None, flow_norm=None, device='cpu', expert_replay=None, env_name=None): #the nones are there so the loading works...
        super().__init__()
        #expert replay is optional for setting the value dice norm paramters
        #could add defaults... or allow empty definition, and fill in things with other functions...     

        #env_name is fix for ant
        
        
        self.flow_type = flow_type # flow type must clearly identiy what model from which source...
        self.flow_args = flow_args
        self.flow_norm = flow_norm # none, sch, vd, ... (for now, not handling sch, due to random...   
        
        #create flow and send to the device
        if self.flow_type in ['MAF','MAFMOG','MADE','MADEMOG']: #add other like real nvp and splines. then give other realvnp a different name
            self.flow = getattr(made, self.flow_type)(**self.flow_args).to(device)
        else:
            raise ValueError("only kamen flows are currently supported here")
        
        if flow_norm == 'vd':
            if expert_replay == None:
                warnings.warn('flow_norm is vd and you didnt set the scale and shift params...')
                self.scale = None
                self.shift = None
            else: 
                scale, shift = get_value_dice_norm(expert_replay, env_name=env_name)
                self.scale = nn.Parameter(scale,requires_grad=False).to(device)
                self.shift = nn.Parameter(shift,requires_grad=False).to(device)
        
        if flow_norm == 'sch':
            self.input_min = None
            self.input_max = None
        
        self.to(device)
        
    def log_prob(self, data):
        #normalize
        data = self.normalize(data)
        return self.flow.log_prob(data)
    
    #(remember, for now we don't use generative side, but in future might need to reverse... (also the normalization transforms the density and we arent really taking that into account (but i guess if q function and actor use it than its fine) (beware of an RL that already normalizes, this will be problematic for our use...)
    
    def normalize(self, data):
        if self.flow_norm == "none":
            return data
        elif self.flow_norm == "vd":
            return (data + self.shift) * self.scale
        elif self.flow_norm == "sch":
            return (data - self.input_min)/(self.input_max - self.input_min) * 2 - 1
    
    
    #add saving and loading...
    #perhaps only needed in the wrapping class...
    #note how double flow will also need that...
    
    
def get_value_dice_norm(expert_replay_buffer, env_name=None): 
    
    shift = torch.from_numpy(np.mean(expert_replay_buffer.data, 0))
    scale = torch.from_numpy(1.0 / (np.std(expert_replay_buffer.data, 0) + 1e-3))

    if env_name == 'Ant-v2':
        if expert_replay_buffer.option == 1:
            scale = torch.cat((scale[:27],scale[-8:]),dim=0) # first 27 dims and last 8 actions
            shift = torch.cat((shift[:27],shift[-8:]),dim=0) # first 27 dims and last 8 actions
        elif expert_replay_buffer.option == 0:
            scale = scale[:27] # first 27 dims of obs
            shift = shift[:27] # first 27 dims of obs
        elif expert_replay_buffer.option == 2:
            scale = torch.cat((scale[:27],scale[111:138]),dim=0) #first 27 of obs and 27 of next obs
            shift = torch.cat((shift[:27],shift[111:138]),dim=0) #first 27 of obs and 27 of next obs

    if env_name == 'Humanoid-v2':
        idx = get_humanoid_obs_dims_array() #get non zero dims
        if expert_replay_buffer.option == 1:
            b = np.array([True]*17) # last 17 action dims
            idx = np.concatenate((idx,b))
            scale = scale[idx]
            shift = shift[idx]
        elif expert_replay_buffer.option == 0:
            scale = scale[idx]
            shift = shift[idx]
        elif expert_replay_buffer.option == 2:
            idx = np.concatenate((idx,idx))
            scale = scale[idx]
            shift = shift[idx]
    
 
    return scale, shift



    