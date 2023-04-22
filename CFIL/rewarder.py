from this import d
import numpy as np
import torch
import torch.nn as nn
import math
from torch.distributions.normal import Normal
import torch.nn.functional as F
import gym
import os
import json

import sys

import utils_data
import utils_flow

import matplotlib.pyplot as plt
from numpy.polynomial.polynomial import polyval

# better graph style
import seaborn as sns
sns.set(style="darkgrid", font_scale=1)


import spinup.algos.pytorch.sac_cfil.sac as sac
import spinup.algos.pytorch.td3_cfil.td3 as td3
import spinup.utils.plot


### -------------rewarder-----------------------

class Rewarder(nn.Module):
    def __init__(self, model, expert_data, update_every=1000, update_iters=10, update_batch_size=100, debug=False):
        super().__init__()
        self.model = model
        self.expert_data = expert_data
        self.option = model.option 
        self.device = next(self.model.parameters()).device

        self.update_every = update_every 
        self.update_iters = update_iters
        self.update_batch_size = update_batch_size


        self.debug=debug

    
    def get_reward(self, batch, not_rl=False):
        #with torch.no_grad(), this will be inside each...
        #for now batch is assumed to be dictionary from spinningup
        if not_rl:
            pass # its already a tensor (assumed with right option)
        elif self.option==1:
            batch = torch.cat((batch['obs'],batch['act']),dim=1)
        elif self.option==0:
            batch = batch['obs']
        elif self.option==2:
            batch = torch.cat((batch['obs'],batch['obs2']),dim=1)
        r = self.model.get_reward(batch)

        return r
    
    def update(self, agent_data, current_timestep=None,RL_start_steps=None,not_rl=False): 

        if not_rl:
            print('updating')
            self.model.update(self.expert_data, agent_data, device=self.device, not_rl=not_rl, iterations=self.update_iters, batch_size=self.update_batch_size)
            print('done updating')  
            return
            
        if current_timestep % self.update_every == 0:
            print('updating')
            self.model.update(self.expert_data, agent_data, device=self.device,iterations=self.update_iters, batch_size=self.update_batch_size)
            print('done updating')          


    #model must have update, get_reward, option. 




### -------------Coupled flow-------------


class CoupledFlow(nn.Module):
    def __init__(self, flow1, flow2, learning_rate, option, device='cpu', use_tanh=False, tanh_scale=(1,1), tanh_shift=False, flow_reg=False, flow_reg_weight=1, smooth=None, env_name=None, rewarder_replay_size=None): 
        super(CoupledFlow, self).__init__()

        self.option = option

        self.env_name = env_name 
        self.rewarder_replay_size = rewarder_replay_size
        
        self.use_tanh = use_tanh
        self.tanh_scale = tanh_scale[0]
        self.tanh_unscale = tanh_scale[1]
        self.tanh_shift = tanh_shift
        self.flow_reg = flow_reg
        self.flow_reg_weight = flow_reg_weight
        self.smooth = smooth
        
        self.flow1 = flow1
        self.flow2 = flow2
        
        self.optimizer = torch.optim.Adam(list(self.flow1.parameters()) + list(self.flow2.parameters()), lr=learning_rate)

        if env_name == 'Humanoid-v2':
            self.humanoid_idx = utils_data.get_humanoid_obs_dims_array()
        
        self.to(device)

    def get_flow1_log_probs(self, data):
        if self.env_name == 'Ant-v2': #-----------------SPECIAL  HANDLING OF ANT DUE TO NO CONTACT FORCES https://www.gymlibrary.ml/environments/mujoco/ant/
            if self.option == 1:
                data = torch.cat((data[:,:27],data[:,-8:]),dim=1) # first 27 dims and last 8 actions
            elif self.option == 0:
                data = data[:,:27] # first 27 dims of obs
            elif self.option == 2:
                data = torch.cat((data[:,:27],data[:,111:138]),dim=1) #first 27 of obs and 27 of next obs
        elif self.env_name == 'Humanoid-v2':
            if self.option == 1:
                b = np.array([True]*17) # last 17 action dims
                idx = np.concatenate((self.humanoid_idx,b))
                data = data[:,idx]
            elif self.option == 0:
                data = data[:,self.humanoid_idx]
            elif self.option == 2:
                idx = np.concatenate((self.humanoid_idx,self.humanoid_idx))
                data = data[:,idx]

        return self.flow1.log_prob(data)
    
    def get_flow2_log_probs(self, data):
        if self.env_name == 'Ant-v2': #-----------------SPECIAL HANDLING OF ANT DUE TO NO CONTACT FORCES https://www.gymlibrary.ml/environments/mujoco/ant/
            if self.option == 1:
                data = torch.cat((data[:,:27],data[:,-8:]),dim=1) # first 27 dims and last 8 actions
            elif self.option == 0:
                data = data[:,:27] # first 27 dims of obs
            elif self.option == 2:
                data = torch.cat((data[:,:27],data[:,111:138]),dim=1) #first 27 of obs and 27 of next obs
        elif self.env_name == 'Humanoid-v2':
            if self.option == 1:
                b = np.array([True]*17) # last 17 action dims
                idx = np.concatenate((self.humanoid_idx,b))
                data = data[:,idx]
            elif self.option == 0:
                data = data[:,self.humanoid_idx]
            elif self.option == 2:
                idx = np.concatenate((self.humanoid_idx,self.humanoid_idx))
                data = data[:,idx]
        
        return self.flow2.log_prob(data)
        
        
    def x(self, data, training=False):
        a = self.get_flow1_log_probs(data)
        b = self.get_flow2_log_probs(data)
        x = a - b

        if self.use_tanh: 
            return self.tanh_unscale*F.tanh(x/self.tanh_scale) 

        return x
        
    def calc_loss(self, p,q): #p is expert data, q is pi data
        #a = self.x(p).exp().mean().log()
        a = torch.logsumexp(self.x(p, training=True), dim=0) - math.log(p.shape[0]) 
        b = self.x(q, training=True).mean()
        loss = a-b
        
        return loss #see value dice equation 7
        
    def get_reward(self, batch):
        # -x is the reward
        with torch.no_grad():
            r = -self.x(batch)

        if self.tanh_shift:
                return r + self.tanh_unscale
            
        return r

    def smoother(self, data):
        if self.smooth:
            return data + (self.smooth)*((data+0.001)*(torch.rand(data.shape).to(data.device)-1/2)) #smooth each dimension of the state with uniform noise scaled to its value
        else:
            return data
   
   
    def update(self, data1, data2, device, iterations=10, batch_size=100, not_rl=False):
        #data1 - expert. data2 - agent.                
        for t in range(iterations):
            ind1 = np.random.randint(0, len(data1), size=batch_size)
            
            batch1 = data1[ind1].to(device)
                        
            if not_rl: #just a bc trajectory
                ind2 = np.random.randint(0, len(data2), size=batch_size)
                batch2 = data2[ind2].to(device)
            else:
                #data2 is spinningup replay buffer
                batch2 = data2.sample_batch(batch_size=batch_size,rewarder=None, limit=self.rewarder_replay_size) #already on device...
                if self.option==1:
                    batch2 = torch.cat((batch2['obs'],batch2['act']),dim=1)
                elif self.option==0:
                    batch2 = batch2['obs']
                elif self.option==2:
                    batch2 = torch.cat((batch2['obs'],batch2['obs2']),dim=1)

            batch1 = self.smoother(batch1)
            batch2 = self.smoother(batch2)
            
            loss = self.calc_loss(batch1, batch2)

            if self.flow_reg:
                losses  = self.get_flow1_log_probs(batch2.to(device))
                loss11 = -losses.mean()

                losses = self.get_flow2_log_probs(batch1.to(device))
                loss22 = -losses.mean()

                loss = loss + (loss11 + loss22)*self.flow_reg_weight

            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()




import spinup.algos.pytorch.sac_cfil.sac as sac
import spinup.algos.pytorch.td3_cfil.td3 as td3
import spinup.utils.plot

    
def plot_save_rl(outputdir, values=None): 
    
    if values== None:
        values = ['Performance',]
    
    spinup.utils.plot.make_plots(all_logdirs=[outputdir], xaxis='TotalEnvInteracts', values=values, save=True)


def save_rewarder(rewarder, outputdir):
    rewarder_dict = {'rewarder_state_dict': rewarder.state_dict(), 'optimizer_state_dict': rewarder.model.optimizer.state_dict(),'option': rewarder.option}

    torch.save(rewarder_dict,outputdir)


def run_rewarder_bc_rl(rewarder, outputdir, env_name, RL_algo='sac', device='cuda:0', seed=0, epochs=40, start_steps=10000):
    
    import copy
    rewarder_copy = copy.deepcopy(rewarder)


    env_fn = lambda : gym.make(env_name)
    logger_kwargs= dict(output_dir=outputdir)

    #might want to add the density output and things like that, to the output...

    #run algorithm (for now, using all the default parameter)
    if RL_algo =='sac':
        sac.sac(env_fn=env_fn, rewarder=rewarder_copy,epochs=epochs, start_steps=start_steps, device=rewarder.device, seed=seed, logger_kwargs=logger_kwargs) #prin...
    elif RL_algo == 'td3':
        td3.td3(env_fn=env_fn, rewarder=rewarder_copy,epochs=epochs, start_steps=start_steps, device=rewarder.device, seed=seed, logger_kwargs=logger_kwargs) #prin...
    
    save_rewarder(rewarder_copy, outputdir + '/rl_rewarder_dict')

    
    plot_save_rl(outputdir)


def run_bc_rl_experiment(rewarder_type, rewarder_args, env_name, data_set_name, flow_type, flow_args, flow_norm, learning_rate, RL_algo='sac', device='cuda:0', seed=0, option=1, num_train_trajs=4, num_test_trajs=1, train_skip_steps=1, outputdir=None, title=None, epochs=40, start_steps=10000, update_every=1000, update_iters=10, update_batch=100):

    config = locals() #for saving info later

    torch.manual_seed(seed)
    np.random.seed(seed)   
    
    state_dim, action_dim = utils_data.get_env_dims(env_name) 
    if option == 1:
        flow_args['input_size'] = state_dim + action_dim
    elif option == 0:
        flow_args['input_size'] = state_dim
    elif option == 2:
        flow_args['input_size'] = state_dim*2

    if env_name == 'Ant-v2':
        if option == 1:
            temp = 27 + action_dim
        elif option == 0:
            temp = 27
        elif option == 2:
            temp = 27*2
        
        flow_args['input_size'] = temp
    elif env_name == 'Humanoid-v2':
        if option == 1:
            temp = 270 + 17
        elif option == 0:
            temp = 270
        elif option == 2:
            temp = 270*2
        
        flow_args['input_size'] = temp
    

    #load data
    expert_replay_buffer, expert_test_data = utils_data.get_train_buffer_and_test_data(data_set_name, env_name, state_dim, action_dim, num_train_trajs, num_test_trajs, device, option, train_skip_steps=train_skip_steps)
    expert_level = utils_data.get_expert_level(data_set_name, env_name)
    
    expert_train_data = torch.from_numpy(expert_replay_buffer.data).to(device)
    
    if rewarder_type == 'coupledflow':     
        flow1 = utils_flow.Flow(flow_type=flow_type,flow_args=flow_args, flow_norm=flow_norm,expert_replay=expert_replay_buffer,env_name=env_name)
        flow2 = utils_flow.Flow(flow_type=flow_type,flow_args=flow_args, flow_norm=flow_norm,expert_replay=expert_replay_buffer,env_name=env_name)
        model = CoupledFlow(flow1, flow2, learning_rate, option, device=device, env_name=env_name, **rewarder_args).to(device) #added env_name for handling ant
    else:
        raise ValueError('enter valid rewarder type')

    rewarder = Rewarder(model=model, expert_data=expert_train_data, update_every=update_every, update_iters=update_iters, update_batch_size=update_batch)
    if outputdir == None:
        outputdir = f'./RESULTS/{rewarder_type}/{env_name}'
    else:
        outputdir = f'./RESULTS/{rewarder_type}/{env_name}/{outputdir}'
    outputdir = utils_data.outputdir_make_and_add(outputdir, title=title) # eventually do this experiment saving differently 
    print(outputdir)
    # move printing to output txt
    sys.stdout = open(outputdir + '/output.txt','wt') # = sys.stderr 
    print(outputdir)
    
    #save general info to the outputdir:
    with open(outputdir+'/general_config', 'w') as f:
        json.dump(config, f)
    
    #run
    run_rewarder_bc_rl(rewarder, outputdir, env_name=env_name, RL_algo=RL_algo, device=device, seed=seed, epochs=epochs, start_steps=start_steps)



if __name__== '__main__':
    # later use argparse
    import argparse
    #see sac look online quickly...
    parser = argparse.ArgumentParser()
    parser.add_argument('--rewarder_type', type=str, default='coupledflow')
    parser.add_argument('--env_name', type=str, default='HalfCheetah-v2')
    parser.add_argument('--data_set_name', type=str, default='Value_Dice')
    parser.add_argument('--num_train_trajs', type=int, default=4)
    parser.add_argument('--num_test_trajs', type=int, default=1) #not even used, nor relevant
    parser.add_argument('--train_skip_steps', type=int, default=1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    
    parser.add_argument('--outputdir', type=str, default=None)
    parser.add_argument('--title', type=str, default=None)
 
    parser.add_argument('--epochs', type=int, default=80) # add as param... above in theory should have all the other spinningup params, but for now wont bother...
    parser.add_argument('--RL_algo', type=str, default='sac') # for now, really the only handled option is sac
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--option', type=int, default=1) # only handled option for now is state_action....
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--start_steps',type=int, default=10000) 
    parser.add_argument('--flow_norm', type=str, default=None)

    parser.add_argument('--flow_reg', action='store_true')
    parser.add_argument('--reg_weight',type=float, default=1)
    parser.add_argument('--smooth',type=float,default=None) #
    
    parser.add_argument('--rewarder_replay_size', type=int, default=None) # if none then entire... one...

    parser.add_argument('--batch_norm', action='store_true') # adding batch norm option for the flow...

    parser.add_argument('--use_tanh', action='store_true')
    parser.add_argument('--tanh_scale', type=float, nargs=2, default=[1,1]) # will be scale and rev_scale
    parser.add_argument('--tanh_shift', action='store_true')

    parser.add_argument('--update_every', type=int, default=1000)
    parser.add_argument('--update_batch', type=int, default=100)
    parser.add_argument('--update_iters', type=int, default=10) #

    args = parser.parse_args()

    args_dict = vars(args)

    #global flow, single layered MAF
    flow_type = 'MAF'
    #input_size is handled inside function later, accroding to option
    flow_args = {"input_size": None, "n_blocks": 1, "hidden_size": 256, "n_hidden": 2, "cond_label_size": None, "activation": "relu", "input_order": "sequential", "batch_norm": args.batch_norm} #currently independent of environment


    if args.flow_norm == None:
        if args.env_name == 'Walker2d-v2':
            flow_norm = 'none' 
        elif args.env_name == 'HalfCheetah-v2':
            flow_norm = 'none'
        elif args.env_name == 'Hopper-v2':
            flow_norm = 'none'
        elif args.env_name == 'Ant-v2':
            flow_norm = 'none'
        elif args.env_name == 'Humanoid-v2':
            flow_norm = 'none'

        args_dict['flow_norm'] = flow_norm
                
    if args.rewarder_type == 'coupledflow':
        rewarder_args = {'smooth':args.smooth, 'use_tanh':args.use_tanh, 'tanh_scale':args.tanh_scale,'tanh_shift':args.tanh_shift,'flow_reg':args.flow_reg, 'flow_reg_weight':args.reg_weight, 'rewarder_replay_size':args.rewarder_replay_size}
    
    del args_dict['smooth']
    del args_dict['flow_reg']
    del args_dict['use_tanh']
    del args_dict['tanh_scale']
    del args_dict['tanh_shift']
    del args_dict['reg_weight']
    del args_dict['rewarder_replay_size']
    del args_dict['batch_norm']
    #Call the function:
    run_bc_rl_experiment(rewarder_args=rewarder_args, flow_type=flow_type, flow_args=flow_args, **args_dict)
    