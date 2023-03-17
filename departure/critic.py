import torch
import torch.nn as nn

from misc import build_net

class Q_Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_dim, action_dim, hid_shape, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            hidden_dim (int): Number of nodes in the network layers

        """
        super(Q_Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        layers = [state_dim + action_dim] + list(hid_shape) + [1]

        self.Q_1 = build_net(layers, nn.ReLU, nn.Identity)
        self.Q_2 = build_net(layers, nn.ReLU, nn.Identity)

    def forward(self, state, action):
        """Build two critic (value) networks that map (state, action) pairs -> Q-values."""
        sa = torch.cat([state, action], 1)
        q1 = self.Q_1(sa)
        q2 = self.Q_2(sa)
        return q1, q2
