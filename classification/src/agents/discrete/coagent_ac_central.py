'''
Critic : Per Layer Critic
Credit Assignment : Boostrapping
'''
# The Actor Critic variant of the code.
import numpy as np
import copy
from src.agents.agent_template import ClassificationAgent
from src.agents.discrete.coagent import Coagent
import torch
import torch.nn as nn
import torch.nn.functional as f
from collections import namedtuple
from src.models.registry import get_model
from src.optimizers.registry import get_optimizer
import time
from icecream import ic

actionPolicy = namedtuple('actionPolicy', ['action', 'action_prob'])
MIN_DENOM = 1e-5

def define_network(in_features, nodes, output, num_layers = 0):
    # define a critic network
    if num_layers == 0:
        model =  nn.Linear(in_features, output)

        # model.weight.data.fill_(0.0)
        # model.bias.data.fill_(0.0)

    else:
        model = nn.Sequential()
        model.add_module("input-layer",nn.Linear(in_features, nodes))
        model.add_module("input-activation", nn.ReLU())
        for i in range(num_layers-1):
            model.add_module(f"{i}-layer",nn.Linear(nodes, nodes))
            model.add_module(f"{i}-activation",nn.ReLU())

        model.add_module(f"{num_layers}-layer",nn.Linear(nodes, output))
    return model

class CoagentACCentral(Coagent):
    '''
    This Agent does not use a critic as of now
    '''
    def __init__(self, params):
        super(CoagentACCentral, self).__init__(params)
        self.critic_layers = params.get('critic_layer', 1)
        self.critic_nodes = params.get('critic_nodes', 1024)
        self.critic_nodes_input = params.get('critic_nodes_input', 1024)
        self.layer_critic = []
        self.layer_critic_optimizers = []
        self.critic_alpha = params.get('critic_alpha_ratio', 2) * self.alpha # the ratio of the learning rates

        for i in range(2):
            if i == 0: # the first layer
                self.layer_critic.append( define_network(params['in_features'][0] + self.num_coagents,self.critic_nodes_input, 1, self.critic_layers ) )
            else:
                self.layer_critic.append( define_network((self.num_coagents * 2) + 1, self.critic_nodes, 1, self.critic_layers)) # "+1" for the layer index 

            self.layer_critic_optimizers.append(get_optimizer(self.optim_type)(self.layer_critic[-1].parameters(), lr  = self.critic_alpha)) # TODO # Q? why use "-1" i.e. most recent appended item to list, rather than "i" as in index here?   


    def train(self, batch_x, batch_y):
        torch.set_printoptions(profile="full") # TEMP 
        # class_probs = self.forward(batch_x)
        class_values, self.coagent_states = self.network(batch_x, greedy = False)

        criterion = nn.CrossEntropyLoss()
        # self.optimizer.zero_grad()
        losses = []
        loss = criterion(class_values, batch_y) # this is the negative reward


        with torch.no_grad():
            delta_loss = nn.CrossEntropyLoss(reduce=False)
            delta = - delta_loss(class_values, batch_y).unsqueeze(1) # use the negative value of loss as reward

        coagent_loss = torch.tensor([0.0 ])
        
        coagent_state_int = [k.long() for k in self.coagent_states]
        # ic(f'XXX {coagent_state_int[0].shape}') # TEMP
        # ic(f'XXX {coagent_state_int[0]}') # TEMP
        # assert 0 # TEMP 


        # start in reverse
        for i in range(self.network.num_coagent_layers() - 1, -1, -1):
            len_batch = batch_x.shape[0] # TODO better 
            index_tensor = torch.ones([len_batch, 1])*i # TODO better 
            if i == 0: 
                data_x = batch_x
                input_x = data_x
            if i != 0:
                data_x = self.coagent_states[i-1]
                input_x = torch.cat((self.coagent_states[i-1], index_tensor), dim= 1) # input for critic  # FIXME check if correct state is selected or now

            with torch.no_grad():
                action = self.coagent_states[i] # get the action
                input_x = torch.cat((input_x, action), dim = 1)

            if i == 0: 
                critic_output = self.layer_critic[0](input_x) # input critic 
            if i != 0: 
                critic_output = self.layer_critic[1](input_x) # global critic 
            # update the critic
            if i == self.network.num_coagent_layers() - 1:
                critic_target = delta
            if i != self.network.num_coagent_layers() - 1:
                with torch.no_grad():
                        state_x_1 = self.coagent_states[i]
                        # action_x_1 = self.coagent_states[i+1] # ON-POLICY
                        action_x_1 = self.network.get_forward_softmax(self.network.layers[i+1], state_x_1).max(dim = 2)[1].float() # OFF-POLICY
                        input_x_1 = torch.cat((state_x_1, index_tensor), dim=1) 
                        input_x_1 = torch.cat((input_x_1, action_x_1), dim=1)
                        critic_target = self.layer_critic[1](input_x_1) # use global critic # self.layer_critic[1]

            mse_loss = nn.MSELoss()
            critic_loss = mse_loss(critic_output, critic_target)
            if i == 0: 
                self.layer_critic_optimizers[0].zero_grad()
            if i != 0: 
                self.layer_critic_optimizers[1].zero_grad()
            critic_loss.backward()
            if i == 0: 
                self.layer_critic_optimizers[0].step()
            if i != 0: 
                self.layer_critic_optimizers[1].step()
            
            with torch.no_grad():
                reward = critic_target

            pi_all = self.network.get_forward_softmax(model = self.network.layers[i], x = data_x)
            mask2 = coagent_state_int[i].long().unsqueeze(2) # TODO investigate if this is the layer number / time step being made part of the input? 
            pi_a = torch.gather(pi_all, 2, mask2).squeeze(2)

            # pi_a = torch.masked_select(pi_all , mask = mask).view(pi_all.shape[0], -1)
            log_pi_a = torch.log(pi_a + MIN_DENOM)
            coagent_loss += (- (log_pi_a * reward).mean(dim = 0) ).sum() # take a mean over the batch

        
        coagent_loss += loss
        self.optimizer.zero_grad()
        coagent_loss.backward()


        if self.gradient_clipping == 'none' or self.gradient_clipping <= 0.0:
            pass
        else:
            nn.utils.clip_grad_norm_(self.network.parameters(), self.gradient_clipping)

        self.optimizer.step()
        losses.append(loss.item())

        

        return losses


