import torch
from torch import nn
import matplotlib.pyplot as plt
import os
import os.path as osp

DISCRETE_ENVS = ['CartPole-v0', 'MountainCar-v0']

# Neural networks for use in BC + DAGGER
class DiscretePolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DiscretePolicy, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, action_dim)
        )
        
    def forward(self, states):
        '''Returns action distribution for all states s in our batch.
        
        :param states: torch.Tensor, size (B, state_dim)
        
        :return logits: action logits, size (B, action_dim)
        '''
        logits = self.net(states)
        return logits.float()

# Plotting utils
def plot_losses(epochs, losses, env_name, is_dagger=True):
    plt.plot(epochs, losses)
    
    if is_dagger:
        plt.title(f'DAGGER losses for {env_name}')
    else:
        plt.title(f'BC losses for {env_name}')
    
    plt.xlabel('epochs')
    plt.ylabel('loss')
    
    if not osp.exists('./plots/bc'):
        os.makedirs('./plots/bc')
    if not osp.exists('./plots/dagger'):
        os.makedirs('./plots/dagger')
    
    plot_dir = osp.join('./plots', 'dagger' if is_dagger else 'bc')
    if is_dagger:
        plt.savefig(osp.join(plot_dir, f'dagger_losses_{env_name}.png'))
    else:
        plt.savefig(osp.join(plot_dir, f'bc_losses_{env_name}.png'))