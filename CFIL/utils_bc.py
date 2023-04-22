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
from utils_data import *
from utils_flow import *

import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval


#will contain the general function for running a bc and also for evaluating...
#and plotting


#determinisic actor
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)
        
        self.max_action = max_action
        

    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))

    
#verify reproducible    
def run_behavioral_cloning(device, data_set_name, env_name, num_train_trajs=10, num_test_trajs=1, seed=0, iterations=15000, batch_size=100, eval_every=10):
    
    env, state_dim, action_dim, max_action = env_setup(env_name, seed)
    
    actor = Actor(state_dim, action_dim, max_action).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
    
    expert_level = get_expert_level(data_set_name, env_name)
    expert_replay_buffer, expert_test_data = get_train_buffer_and_test_data(data_set_name, env_name, state_dim, action_dim, num_train_trajs, num_test_trajs, device, option=1, train_skip_steps=1) #option is not important here
    test_s = torch.from_numpy(expert_test_data[:,:state_dim]).to(device)
    test_a = torch.from_numpy(expert_test_data[:,state_dim:]).to(device)
    
    iterations = iterations
    batch_size = batch_size
    eval_every = eval_every
    
    
    train_losses = []
    test_losses = []
    reward_avges = [] 
    reward_sums = [] 
    ep_lens = []
    trajectories = []
    
    index = 0

    for i in range(iterations):
        state, action, next_state, reward2, not_done, _ = expert_replay_buffer.sample(batch_size)
        actor_bc_loss = F.mse_loss(actor(state), action)

        optimizer.zero_grad()
        actor_bc_loss.backward()
        optimizer.step()

        if i % eval_every == 0:

            #env eval and expert_density:
            #generate actor trajectory: (maybe generate more than 1?) (also maybe add eval step before first update)
            #maybe generate in completion to 1000 so hopper results are more accurate?
            state = env.reset()
            states_actions = []
            done = False
            cum_return = 0
            ep_len = 0
            while not done:
                state = torch.FloatTensor(state.reshape(1, -1))
                with torch.no_grad():
                    action = actor(state.to(device)).cpu().data.numpy().flatten()
                next_state, reward, done, _ = env.step(action)
                cum_return += reward
                ep_len += 1
                states_actions.append(np.concatenate((state.reshape(-1),action),axis=0))

                state = next_state

            #states = torch.tensor(states,dtype=torch.float32).to(device)

            test_loss = F.mse_loss(actor(test_s), test_a).item()

            train_losses.append(actor_bc_loss.item())
            test_losses.append(test_loss)
            reward_avges.append(cum_return/ep_len) 
            reward_sums.append(cum_return)
            ep_lens.append(ep_len)
            trajectories.append(np.stack(states_actions)) #make sure np.stack is doing what you expect

            print(f"iter: {i}, train: {actor_bc_loss.item():.4f}, test: {test_loss:.4f}, return: {cum_return:.4f}")

            index += 1
    else:
        print('-------------------------------')
        print('DONE, now saving general data...', end='')
        #save general bc data
        #create outputdir #***need to improve the folder name
        outputdir = 'RESULTS/BC_general/' + env_name
        folder_name = '/data=' + data_set_name + '_traj=' + str(num_train_trajs) + '_eval=' + str(eval_every) #something that includes most info (data set name, num_trajectories, batch, iter,_eval..... and so on)
        outputdir += folder_name
        os.makedirs(outputdir,exist_ok=True)

        #save params
        params = {'device':str(device), 'data_set_name':data_set_name,
                  'env_name':env_name, 'num_train_trajs':num_train_trajs,
                  'num_test_trajs':num_test_trajs, 'seed':seed,
                  'iterations':iterations, 'batch_size':batch_size,
                  'eval_every':eval_every, 'expert_level':expert_level}

        with open(outputdir+'/general_params', 'w') as f:
            json.dump(params, f)  

        #save lists
        training_info = {'train_losses' : train_losses, 'test_losses' : test_losses,
                         'reward_avges' : reward_avges, 'reward_sums': reward_sums,
                         'ep_lens':ep_lens,'trajectories':trajectories}
        torch.save(training_info, outputdir + '/eval_data_and_trajecories.pt')
        
        #potentially have separate function for saving bc info, since this is getting ugly

        print('DONE')
        print('-------------------------------')

        
        #plot train and reward
        plt_bc_eval_info(outputdir, save=True, expert_level=expert_level)
        
        return outputdir
        

def load_bc_trajectories(path): #example path should be 'RESULTS/BC_general/Hopper-v2/data=Value_Dice_traj=4_eval=10'
    
    #add option for env and data set input, add a default path, since this will be pretty much only used with the same
    
    
    training_info = torch.load(path + '/eval_data_and_trajecories.pt') 
    
    print('Trajecories loaded.\nRemember they are stored as list of numpy arrays with states and actions concatenated. Also, remember eval_every.')
    
    return training_info['trajectories']



def plt_bc_eval_info(path, save=False, expert_level=None): #potentially bad that im loading also trajectories just for the other info so maybe separate them
    training_info = torch.load(path + '/eval_data_and_trajecories.pt') 
    
    # add expert_level 
    
    plt.title("bc train and test")
    plt.plot(training_info['train_losses'], label="train_loss")
    plt.plot(training_info['test_losses'], label="test_loss")

    plt.legend()

    if save:
        plt.savefig(path + '/bc_train_test')
        
        
    fig, axs = plt.subplots()
    axs.set_title("bc reward every eval iteration")
    axs.plot(training_info['reward_sums'], label="reward")
    
    if expert_level:
        axs.axhline(expert_level, label="expert level", color="r", linestyle="dashed")

    axs.legend()
    if save:
        plt.savefig(path + '/bc_reward_every_eval')
