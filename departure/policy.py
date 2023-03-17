import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from misc import build_net

class Policy(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_dim, action_dim, hid_shape, seed, h_acti=nn.ReLU, o_acti=nn.ReLU, log_std_min=-20, log_std_max=2, device='cpu'):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hid_shape (tuple): Number of nodes in hidden layers
        """
        super(Policy, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        layers = [state_dim] + list(hid_shape)
        self.a_net = build_net(layers, h_acti, o_acti)

        self.mu_layers = [nn.Sequential(nn.Linear(layers[-1], layers[-1]), 
                                        h_acti(),  
                                        nn.Linear(layers[-1], 1)
                                        ).to(device)
                                        for _ in range(action_dim)]
        self.log_std_layers = [nn.Sequential(nn.Linear(layers[-1], layers[-1]), 
                                        h_acti(),  
                                        nn.Linear(layers[-1], 1)
                                        ).to(device)
                                        for _ in range(action_dim)]

    def forward(self, state, deterministic=False, with_logprob=True):
        '''Network with Enforcing Action Bounds'''
        # print(state)
        net_out = self.a_net(state)
        mu = torch.cat([mu_layer(net_out) for mu_layer in self.mu_layers], 1)
        log_std = torch.cat([log_std_layer(net_out) for log_std_layer in self.log_std_layers], 1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        dist = Normal(mu, std)

        if deterministic: u = mu
        else: u = dist.rsample() #'''reparameterization trick of Gaussian'''#
        a = torch.tanh(u)

        if with_logprob:
            # get probability density of logp_pi_a from probability density of u, which is given by the original paper.
            # logp_pi_a = (dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)).sum(dim=1, keepdim=True)
            # Derive from the above equation. No a, thus no tanh(h), thus less gradient vanish and more stable.
            logp_pi_a = dist.log_prob(u).sum(axis=1, keepdim=True) - (2 * (np.log(2) - u - F.softplus(-2 * u))).sum(axis=1, keepdim=True)
        else:
            logp_pi_a = None

        return a, logp_pi_a
    
    def select_action(self, state, deterministic):
		# only used when interact with the env
        with torch.no_grad():
            state = torch.FloatTensor(state.reshape(1, -1)).to(next(self.parameters()).device)
            a, _ = self.forward(state, deterministic, with_logprob=False)
        return a.cpu().numpy().flatten()
     