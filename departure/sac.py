import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributed.rpc import rpc_async
import numpy as np
import time

from misc import soft_update, hard_update, enable_gradients, disable_gradients, _call_method, _remote_method
from critic import Q_Critic
from worker import Worker
from replay_buffer import PrioritizedReplay
from parameter_server import ParameterServer
from policy import Policy

MSELoss = torch.nn.MSELoss()

class SAC(object):
    """
    Wrapper class for SAC agents with central attention critic in multi-agent
    task
    """
    def __init__(self, env, b_rref, p_rref, config):
        """
        Inputs:
            agent_init_params (list of dict): List of dicts with parameters to
                                              initialize each agent
                num_in_pol (int): Input dimensions to policy
                num_out_pol (int): Output dimensions to policy
            sa_size (list of (int, int)): Size of state and action space for
                                          each agent
            gamma (float): Discount factor
            tau (float): Target update rate
            pi_lr (float): Learning rate for policy
            q_lr (float): Learning rate for critic
            reward_scale (float): Scaling for reward (has effect of optimal
                                  policy entropy)
            hidden_dim (int): Number of hidden dimensions for networks
        """
        
        self.b_rref = b_rref
        self.p_rref = p_rref
        
        # Policy
        self.policy = Policy(env.state_dim, env.action_dim, config.hid_shape, config.seed, device=config.device).to(config.device)
        p_rref and _remote_method(ParameterServer.push, p_rref, {k: v.cpu() for k, v in self.policy.state_dict().items()})
        self.target_policy = Policy(env.state_dim, env.action_dim, config.hid_shape, config.seed, device=config.device).to(config.device)
        hard_update(self.target_policy, self.policy)
        self.policy_optimizer = Adam(self.policy.parameters(), lr=config.pi_lr)
        
        # Critic
        self.critic = Q_Critic(env.state_dim, env.action_dim, config.hid_shape, config.seed).to(config.device)
        self.target_critic = Q_Critic(env.state_dim, env.action_dim, config.hid_shape, config.seed).to(config.device)
        hard_update(self.target_critic, self.critic)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=config.q_lr, weight_decay=1e-5)
        
        self.config = config
        self.alpha = config.alpha
        self.adaptive_alpha = config.adaptive_alpha
        if config.adaptive_alpha:
            # Target Entropy = −dim(A) (e.g. , -6 for HalfCheetah-v2) as given in the paper
            self.target_entropy = torch.tensor(-env.action_dim, dtype=float, requires_grad=True, device=config.device)
            # We learn log_alpha instead of alpha to ensure exp(log_alpha)=alpha>0
            self.log_alpha = torch.tensor(np.log(config.alpha), dtype=float, requires_grad=True, device=config.device)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=config.q_lr)


    def update(self, experiences):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Q_targets = r + γ * (min_critic_target(next_state, actor_target(next_state)) - α *log_pi(next_action|next_state))
        Critic_loss = MSE(Q, Q_target)
        Actor_loss = α * log_pi(a|s) - Q(s,a)
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
        """
        s, a, r, s_prime, deads, idx, weights = experiences
        s       = torch.FloatTensor(np.float32(s)).to(self.config.device)
        a       = torch.FloatTensor(a).to(self.config.device)
        r       = torch.FloatTensor(r).to(self.config.device).unsqueeze(1) 
        s_prime = torch.FloatTensor(np.float32(s_prime)).to(self.config.device)
        deads   = torch.FloatTensor(deads).to(self.config.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).to(self.config.device).unsqueeze(1)
        # print(s, a, r, s_prime, deads, idx, weights)

        #----------------------------- ↓↓↓↓↓ Update Q Net ↓↓↓↓↓ ------------------------------#
        # Compute Q targets for current states (y_i)
        with torch.no_grad():
            # print(s_prime)
            a_prime, log_pi_a_prime = self.policy(s_prime)
            target_Q1, target_Q2 = self.target_critic(s_prime, a_prime)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = r + (1 - deads) * self.config.gamma * (target_Q - self.alpha * log_pi_a_prime)
                        
        # Get current Q estimates
        current_Q1, current_Q2 = self.critic(s, a)
        q_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        q_loss.backward()
        self.critic_optimizer.step()
        
        td_error1 = target_Q-current_Q1
        td_error2 = target_Q-current_Q2
        prios = abs(((td_error1 + td_error2)/2.0 + 1e-5).squeeze())

        _remote_method(PrioritizedReplay.update_priorities, self.b_rref, idx, prios.data.cpu().numpy())

        #----------------------------- ↓↓↓↓↓ Update Actor Net ↓↓↓↓↓ ------------------------------#
        for params in self.critic.parameters():
            params.requires_grad = 	False

        a, log_pi_a = self.policy(s)
        current_Q1, current_Q2 = self.critic(s, a)
        Q = torch.min(current_Q1, current_Q2)
        actor_loss = ((self.alpha * log_pi_a - Q)*weights).mean()

        # Minimize the loss
        disable_gradients(self.critic)
        self.policy_optimizer.zero_grad()
        actor_loss.backward()
        self.policy_optimizer.step()
        enable_gradients(self.critic)

        #----------------------------- ↓↓↓↓↓ Update alpha ↓↓↓↓↓ -----------------------------------#
        if self.adaptive_alpha:
            # we optimize log_alpha instead of aplha, which is aimed to force alpha = exp(log_alpha)> 0
            # if we optimize aplpha directly, alpha might be < 0, which will lead to minimun entropy.
            alpha_loss = -(self.log_alpha * (log_pi_a + self.target_entropy).detach()).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
            self.alpha = self.log_alpha.exp()
                        
        #----------------------------- ↓↓↓↓↓ Update Target Net ↓↓↓↓↓ ------------------------------#
        soft_update(self.target_critic, self.critic, self.config.tau)
        soft_update(self.target_policy, self.policy, self.config.tau)

    def learn(self):
        train_step = 1
        count_down = self.config.learn_after_rollout # number of training after all workers finish their jobs
        count_unchanged = 0
        n_rollouts = 0

        while True:
            n_rollouts = _remote_method(PrioritizedReplay.get_counts, self.b_rref)
            if n_rollouts > self.config.start_learning_after:
                break
            time.sleep(3)

        K = 40
        eta_t = 0.996
        while count_down > 0:
            for k in range(1,K):
                len = _remote_method(PrioritizedReplay.__len__, self.b_rref)
                c_k = max(int(len*eta_t**(k*(300/K))), 2500)
                sample = _remote_method(PrioritizedReplay.sample, self.b_rref, self.config.batch_size, c_k)
                
                if not bool(sample):
                    time.sleep(10)
                    continue
                
                self.update(sample)
                time.sleep(0.1)

            if train_step % self.config.update_freq == 0:
                _remote_method(ParameterServer.push, self.p_rref, {k: v.cpu() for k, v in self.policy.state_dict().items()})

            train_step += 1
            # condition to stop learning
            last_rollouts = n_rollouts
            n_rollouts = _remote_method(PrioritizedReplay.get_counts, self.b_rref)
            if n_rollouts == last_rollouts:
                count_unchanged += 1
            else:
                count_unchanged = 0
            
            if count_unchanged > 10:
                count_down -= 1
            

    def save(self,episode):
        torch.save(self.policy.state_dict(), f"./model/{self.config.model_name}/actor_{episode}.pth")
        torch.save(self.critic.state_dict(), f"./model/{self.config.model_name}/critic_{episode}.pth")


    def load(self,episode):
        self.policy.load_state_dict(torch.load(f"./model/{self.config.model_name}/actor_{episode}.pth"))
        self.critic.load_state_dict(torch.load(f"./model/{self.config.model_name}/critic_{episode}.pth"))

    @classmethod
    def init_for_testing(cls, env, config):
        instance = cls(env, None, None, config)
        instance.policy.load_state_dict(torch.load(f"./model/{config.model_name}/actor_{config.model_index}.pth"))
        return instance

    def evaluate(self, env, write_csv=False):
        env = env()
        s, done, ep_r = env.reset(write_csv=write_csv), False, 0
        while not done:
            # Take deterministic actions at test time
            a = self.policy.select_action(s, deterministic=True)
            s_prime, r, done, _, _ = env.step(a)
            print(s, a, r, s_prime)
            ep_r += r
            s = s_prime
        return ep_r
    