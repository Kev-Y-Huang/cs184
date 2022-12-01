from common import *
import torch
from torch import optim
import numpy as np
from dataset import ExpertData
from torch import distributions as pyd
import torch.nn as nn
import os
from stable_baselines3 import DQN

'''Imitation learning agents file.'''

class BC:
    def __init__(self, state_dim, action_dim, args):
        # Policy network setup
        self.policy = DiscretePolicy(state_dim, action_dim)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        self.loss = nn.CrossEntropyLoss()

    def get_logits(self, states):
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states)
        return self.policy(states)


    def learn(self, expert_states, expert_actions):
        # TODO Do gradient descent here.
        pass

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

class DAGGER:
    '''TBH can make a subclass of BC learner'''
    def __init__(self, state_dim, action_dim, args):
        self.policy = DiscretePolicy(state_dim, action_dim)
        
        self.optimizer = optim.Adam(self.policy.parameters(), lr=args.lr)
        
        policy_path = os.path.join(args.expert_save_path, args.env.lower() + '_policy.pt')
        self.expert_policy = DQN.load(policy_path)
        self.loss = nn.CrossEntropyLoss()

    def rollout(self, env, num_steps):
        # TODO Rollout for 'num_steps' steps in the environment. Reset if necessary.
        pass
    
    def get_logits(self, states):
        return self.policy(states)

    def sample_from_logits(self, logits):
        # TODO Given logits from our neural network, sample an action from the distribution defined by said logits.
        pass

    def learn(self, expert_states, expert_actions):
        # TODO Gradient descent here, like in BC
        pass
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
    