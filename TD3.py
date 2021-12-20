import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import deque


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Actor(nn.Module):
    
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        self.layer_1 = nn.Linear(state_dim, 400)
        self.layer_2 = nn.Linear(400,300)
        self.layer_3 = nn.Linear(300, action_dim)
        self.max_action = max_action
        
    def forward(self, x):
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        x = self.max_action*torch.tanh(self.layer_3(x))
        return x
    
class Critic(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        #Define the first Critic neural networks
        self.layer_1 = nn.Linear(state_dim + action_dim, 400)
        self.layer_2 = nn.Linear(400, 300)
        self.layer_3 = nn.Linear(300, 1)
        #Defines the second Critic neural network
        self.layer_4 = nn.Linear(state_dim + action_dim, 400)
        self.layer_5 = nn.Linear(400, 300)
        self.layer_6 = nn.Linear(300,1)
        
    def forward(self,x, u):
        xu = torch.cat([x, u], 1)
        #The first Critic Neural Network
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        # The second Critic Neural Networks
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1, x2
    def Q1(self, x, u):
        xu = torch.cat([x, u], 1)
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1
    
#Building the Training Process
class TD3(object):
    
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters())
        self.critic = Critic(state_dim, action_dim).to(device)
        self.critic_target = Critic(state_dim, action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action
        
    def select_action(self, state):
        state = torch.Tensor(state.reshape(1,-1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()
    
    def train(self, replay_buffer, iterations, batch_size =100, discount= 0.99, tau=0.005, policy_noise =0.2, noise_clip =0.5, policy_freq=2):
        
        for it in range(iterations):
            
            # Sample a batch of transitions (s, s', a, r) from the memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device)
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            
            #The Actor target provides the next action a' from the next state s'
            next_action = self.actor_target(next_state)
            
            # Adds Gaussian noise to the next action a' and clamps it in a range of values supported by the environment
            noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
            noise = noise.clamp(-noise_clip, noise_clip)
            next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
            
            # The two Critic targets take each the couple(s', a') as input and return two Q-value Qt1(s', a') and Qt2(s', a') as outputs
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            
            # Choose the minimum of these two Q-values: min (Qt1, Qt2)
            target_Q = torch.min(target_Q1, target_Q2)
            
            # Obtain the final target of the two Critic models: Qt= r+ (\gamma)*min(Qt1, Qt2), where (\gamma) is the dicount factor
            # Detach the target_Q to cut the gradient computation
            target_Q = reward +((1 - done)*discount*target_Q).detach()
            
            # The two Critic models have each the couple (s, a) as in an input and return the two Q-values Q1(s,a) and Q2(s,a) as an outputs
            current_Q1, current_Q2 = self.critic(state, action)
            
            # Compute the loss from the two Critic models: Critic_Loss = Mean Squre Error, MSE_Loss(Q1(s,a),Qt)+MSE_Loss(Q2(s,a),Qt)
            critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
            
            # Backpropagate the Critic loss and update the parameters of the two Critic models with a SGD optimizer
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()
            
            #Delay policy updates(i.e., time scale)
            if it % policy_freq == 0:
                
                # Compute actor loss
                actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
                # Optimize the actor
                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()
                
                # Update the frozen target models for actors
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
                    
                # Update the weights of the critic target by polyak averaging
                
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
            
    #Save a trained model
    def save(self, filename, directory):
        torch.save(self.actor.state_dict(), '%s/%s_actor.pth' % (directory, filename))
        torch.save(self.critic.state_dict(), '%s/%s_critic.pth' % (directory, filename))
            
    #Load a pre-trained model
    def load(self, filename, directory):
        self.actor.load_state_dict(torch.load('%s/%s_actor.pth' % (directory,filename)))
        self.critic.load_state_dict(torch.load('%s/%s_critic.pth' % (directory, filename)))
                
                
    