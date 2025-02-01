import torch
import torch.nn as nn

from misc import build_net

class Q_Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_dim, action_dim, hid_shape, seed):
        """
        Initialize the Critic model.
        
        Parameters:
        - state_dim (int): Dimension of the state space
        - action_dim (int): Dimension of the action space
        - hid_shape (list): List defining the number of nodes in each hidden layer
        - seed (int): Random seed for reproducibility
        """
        super(Q_Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        layers = [state_dim + action_dim] + list(hid_shape) + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        """
        Forward pass through the Critic model.
        
        Parameters:
        - state (Tensor): The state input
        - action (Tensor): The action input
        
        Returns:
        - Tensor: Q-values for the (state, action) pairs
        """
        sa = torch.cat([state, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2
