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
        self.optimizer.zero_grad()
        logits = self.get_logits(expert_states)
        loss = self.loss(logits, expert_actions)
        loss.backward()
        self.optimizer.step()

        return loss.item()

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
        self.expert_policy = DQN.load(policy_path, custom_objects={'learning_rate': 0.0, 'lr_schedule': lambda _: 0.0, 'exploration_schedule': lambda _: 0.0})
        self.loss = nn.CrossEntropyLoss()

    def rollout(self, env, num_steps):
        # State to keep track of datapoints for rollout
        states = []
        expert_actions = []

        state = env.reset()

        for _ in range(num_steps):
            # Append current state in the right format
            state = torch.from_numpy(state).float()
            states.append(state)

            # Get logits based on state and generate the actions from both the current and expert policy
            logits = self.get_logits(state)
            expert_actions.append(torch.from_numpy(self.expert_policy.predict(state, deterministic=True)[0]))
            action = self.sample_from_logits(logits)
            state, _, done, _ = env.step(action)

            # Reset environment if if environment is done
            if done:
                state = env.reset()

        # Return correctly formatted data
        return ExpertData(torch.stack(states), torch.stack(expert_actions))
    
    def get_logits(self, states):
        return self.policy(states)

    def sample_from_logits(self, logits):
        return pyd.categorical.Categorical(logits=logits).sample().numpy()

    def learn(self, expert_states, expert_actions):
        self.optimizer.zero_grad()
        logits = self.get_logits(expert_states)
        loss = self.loss(logits, expert_actions)
        loss.backward()
        self.optimizer.step()

        return loss.item()
    
    def load(self, path):
        self.policy.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.policy.state_dict(), path)
    