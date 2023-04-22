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
#import made #kamen flows

import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval


#Replay Buffer (originally from author's TD3)
class ReplayBuffer(object):
    def __init__(self,state_dim, action_dim, device, option=0, max_size=int(1e6)):
        #option 0 - only states. option 2 - state next-state
        self.option = option
        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))
        
        #data holds whatever is inputed into the flows to calculate the reward
        if option == 0:
            self.data = np.zeros((max_size, state_dim))
        if option == 1:
            self.data = np.zeros((max_size, state_dim + action_dim))
        if option == 2:
            self.data = np.zeros((max_size, 2*state_dim))

        self.device = device


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done
        
        if self.option == 0:
            self.data[self.ptr] = state
        if self.option == 1:
            self.data[self.ptr] = np.concatenate((state,action))
        if self.option == 2:
            self.data[self.ptr] = np.concatenate((state,next_state))

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        


    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            torch.FloatTensor(self.data[ind]).to(self.device)
        )
    
    
def env_setup(env_name, seed, return_env_only = False):
    #setup environment seed
    
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] 
    max_action = float(env.action_space.high[0])
    env.seed(seed)
    env.action_space.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    if return_env_only:
        return env
    else:
        return env, state_dim, action_dim, max_action


def outputdir_make_and_add(outputdir, title=None):
    #creates outputdir
    os.makedirs(outputdir,exist_ok=True)
    folder_num = len(next(os.walk(outputdir))[1]) #counts how many folders already there 
    if folder_num == 0:
        folder_num = 1
    elif folder_num == 1 and next(os.walk(outputdir))[1][0][0] == ".":
        folder_num = 1
    else:
        folder_num = max([int(i.split('-')[0]) for i in next(os.walk(outputdir))[1] if i[0] != '.'],default=0) + 1 # this looks for max folder num and adds one... this works even if title is used (because we index at 1) (dot check to ignore .ipynb) 
        #currently returns error when a subfolder contains anything other than a number (exept dot handle) 
        #so essentially this assumes the outputdir structure with numbers (and possible titles). will need to fix if i want to use it later for something else
        
    if title == None:
        outputdir += '/' + str(folder_num) #adds one
    else:
        outputdir += '/' + str(folder_num) + f'-({title})' #adds one and appends title
    os.makedirs(outputdir,exist_ok=True)
    return outputdir


def get_train_buffer_and_test_data(data_set_name, env_name, state_dim, action_dim, num_train, num_test, device, option, train_skip_steps=1):
    #return train and test datasets. train as buffer and test as data
    train_buffer = load_expert_data(data_set_name, env_name, state_dim, action_dim, device, option=option, num_trajs=num_train, skip_steps=train_skip_steps)
    
    test_buffer = load_expert_data(data_set_name, env_name, state_dim, action_dim, device, option=option, num_trajs=(num_train + num_test), skip_steps=1) #test skipsteps is 1
    test_data = test_buffer.data[-1000*num_test:]
    
    #stupid way to do it but fine for now...
    
    return train_buffer, test_data #test is numpy array, while train is a buffer
    
    
       
def load_expert_data(expert_data_name, env_name, state_dim, action_dim, device, option=0, num_trajs=1, skip_steps=1):
    
    if expert_data_name == "Value_Dice":
        return load_value_dice_data(env_name, state_dim, action_dim, device, option, num_trajs, skip_steps)
    if expert_data_name == "Spinningup":
        return load_spinningup_data(env_name, state_dim, action_dim, device, option, num_trajs, skip_steps)


########specific dataset loading functions should support options and return the data into a replay from above

#this loads the value dice datasets and returns a replay buffer filled with them   
def load_value_dice_data(env_name, state_dim, action_dim, device, option=0, num_trajs=1, skip_steps=1):
    #expert data file path
    env_name = env_name[:-1] + "2" #needed because the value dice data was originaly on v2 environments and those are the file names
    file_path = f"./expert_datasets/value_dice_data/{env_name}.npz"
    
    
    with open(file_path, "rb") as file:
        expert_data = np.load(file)
        expert_data = {key: expert_data[key] for key in expert_data.files}
    
        
        expert_states = expert_data['states']
        expert_actions = expert_data['actions']
        expert_next_states = expert_data['next_states']
        expert_dones = expert_data['dones']
        
        indexes = np.arange(0,1000,skip_steps)
        indexes = np.concatenate([indexes + 1000*i for i in range(num_trajs)])

        
        # split into trajectories (len 1000) than sample according to requested skip steps
        expert_states = expert_states[indexes]
        expert_actions = expert_actions[indexes]
        expert_next_states = expert_next_states[indexes]
        expert_dones = expert_dones[indexes]
        
    expert_replay_buffer = ReplayBuffer(state_dim, action_dim, device, option, max_size=len(expert_states))
    expert_replay_buffer.state = expert_states
    expert_replay_buffer.action = expert_actions
    expert_replay_buffer.next_state = expert_next_states
    expert_replay_buffer.not_done = expert_dones.reshape(-1,1)
    expert_replay_buffer.size = len(expert_states)
    
    if option == 0:
        expert_replay_buffer.data = expert_states
    if option == 1:
        expert_replay_buffer.data = np.concatenate((expert_states,expert_actions), axis=1)
    if option == 2:
        expert_replay_buffer.data = np.concatenate((expert_states,expert_next_states), axis=1)
    
    
    return expert_replay_buffer

#this loads the spinningup datasets and returns a replay buffer filled with them   
def load_spinningup_data(env_name, state_dim, action_dim, device, option=0, num_trajs=1, skip_steps=1):
    #expert data file path
    env_name = env_name[:-1] + "2" #needed because the value dice data was originaly on v2 environments and those are the file names
    file_path = f"./expert_datasets/spinningup_data/{env_name}.npz"
    
    
    with open(file_path, "rb") as file:
        expert_data = np.load(file)
        expert_data = {key: expert_data[key] for key in expert_data.files}
    
        
        expert_states = expert_data['states']
        expert_actions = expert_data['actions']
        expert_next_states = expert_data['next_states']
        expert_dones = expert_data['dones']
        
        indexes = np.arange(0,1000,skip_steps)
        indexes = np.concatenate([indexes + 1000*i for i in range(num_trajs)])

        
        # split into trajectories (len 1000) than sample according to requested skip steps
        expert_states = expert_states[indexes]
        expert_actions = expert_actions[indexes]
        expert_next_states = expert_next_states[indexes]
        expert_dones = expert_dones[indexes]
        
    expert_replay_buffer = ReplayBuffer(state_dim, action_dim, device, option, max_size=len(expert_states))
    expert_replay_buffer.state = expert_states
    expert_replay_buffer.action = expert_actions
    expert_replay_buffer.next_state = expert_next_states
    expert_replay_buffer.not_done = expert_dones.reshape(-1,1)
    expert_replay_buffer.size = len(expert_states)
    
    if option == 0:
        expert_replay_buffer.data = expert_states
    if option == 1:
        expert_replay_buffer.data = np.concatenate((expert_states,expert_actions), axis=1)
    if option == 2:
        expert_replay_buffer.data = np.concatenate((expert_states,expert_next_states), axis=1)
    
    
    return expert_replay_buffer
    
#returns expert level 
def get_expert_level(expert_data_name,env_name):
    expert_level = 0 #expert reward
    
    #hardcoded for valuedice, since the trajectories don't contain rewards [add the source]
    if expert_data_name == "Value_Dice":
        if "Hopper" in env_name:
            expert_level = 3571
        elif "Half" in env_name:
            expert_level = 4463
        elif "Walker" in env_name:
            expert_level = 6717
        elif "Ant" in env_name:
            expert_level = 4228
    if expert_data_name == 'Spinningup':
        if 'Human' in env_name:
            expert_level = 5188
    
    return expert_level


def get_env_dims(env_name):
    #hardcoded to seed interference
    #state and action dim
    if "Hopper" in env_name:
        return 11, 3
    elif "Half" in env_name:
        return 17, 6
    elif "Walker" in env_name:
        return 17, 6
    elif "Ant" in env_name:
        return 111, 8
    elif "Human" in env_name:
        return 376, 17
    elif 'Swim' in env_name:
        return 8, 2
    



#fills  and returns a random replay buffer to size 
#needs to support option (currently only supports option=1)
def get_random_data(env, size, option,state_dim,action_dim,device):
    
    if option != 1:
        raise ValueError("only option = 1 (state+action) is currently supported here in random data")
        
    random_replay_buffer = ReplayBuffer(state_dim, action_dim, device, option=option, max_size=size)

    #fill random data
    state, done = env.reset(), False

    for t in range(size):
        t += 1

        if not done:
            action = env.action_space.sample()
            next_state, reward, done, _ = env.step(action)

            random_replay_buffer.add(state, action, next_state, reward, done)

            state = next_state

        state, done = env.reset(), False
        
    return random_replay_buffer




def get_humanoid_obs_dims_array(): #removal of contact forces https://www.gymlibrary.ml/environments/mujoco/
    return np.array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    False, False, False, False, False, False, False, False, False,
    False,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True, False, False, False, False,
    False, False,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True, False,
    False, False, False, False, False,  True,  True,  True,  True,
    True,  True,  True,  True,  True,  True,  True,  True,  True,
    True,  True,  True,  True, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False, False, False,
    False, False, False, False, False, False, False])