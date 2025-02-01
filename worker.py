import torch.distributed.rpc as rpc
import numpy as np
import torch
import time

from misc import _remote_method
from policy import Policy
from replay_buffer import PrioritizedReplay
from parameter_server import ParameterServer

class Worker:
    r"""
    A worker (or an observer) has exclusive access to its own environment. Each worker
    captures the state of all agents from its environment, and send the state to the master coordinator 
    to select a set of cooperative action. Then, the worker applies the action to its environment
    and reports the rewards to the master coordinator.
    It is highly beneficial for computationally expensive environments.
    """

    def __init__(self, env, p_rref, b_rref, config):
        """
        Initialize the Worker.
        
        Parameters:
        - env (callable): A callable that returns an environment instance
        - p_rref (RRef): A reference to the parameter server
        - b_rref (RRef): A reference to the replay buffer
        - config (arg parser): Configuration object containing various settings
        """
        self.id = rpc.get_worker_info().id
        self.local_policy = Policy(env.state_dim, env.action_dim, hid_shape=config.hid_shape, seed=config.seed)
        self.env = env(config)
        self.p_rref = p_rref
        self.b_rref = b_rref
        self.env.seed(config.seed)
        self.config = config

    def rollout(self):
        """Collect experience using an exploration policy"""
        for ep in range(self.config.n_episodes):
            deterministic = True if (ep+1) % self.config.eval_interval == 0 else False
            while True:
                values = _remote_method(ParameterServer.pull, self.p_rref)
                if values is not None:
                    break
                time.sleep(0.5)

            self.local_policy.load_state_dict(values)
            state, ep_reward, done = self.env.reset(), 0, False
            operational_cost, noise_cost, nfz_penalty = 0, 0, 0

            while not done:
                # Draw action from the distribution given by the actor
                if ep >= self.config.random_exploration_episodes:
                    state = torch.from_numpy(state).float()
                    action = self.local_policy.select_action(state, deterministic=deterministic)
                else:
                    action = np.random.uniform(-0.999, 1, self.env.action_dim)

                # Get environment's reponse
                next_state, reward, done, dead, info= self.env.step(action)
                ep_reward += reward
                operational_cost += info[1]
                noise_cost += info[2]
                nfz_penalty += info[3]

                _remote_method(PrioritizedReplay.store, self.b_rref, state, action, reward, next_state, dead)
                state = next_state

            # End one episode
            print(f'Worker: {self.id-2:2d}\tEpisode: {ep+1:5d}\tReward:\t{ep_reward:8.2f}\tFinal position: {state[0] + self.env.initial_lat - self.env.target_lat:7.4f} {state[1]+ self.env.initial_lon - self.env.target_lon:7.4f} {state[2]:7.4f} {state[6]:7.4f}\t\tTime spent: {info[0]}', flush=True)
            # print(f'Worker: {self.id-2:2d}\tEpisode: {ep+1:5d}\tReward:\t{ep_reward:8.2f}\tFinal position:', flush=True)
            _remote_method(ParameterServer.write, self.p_rref, self.id-2, ep+1, ep_reward, operational_cost, noise_cost, nfz_penalty)
            if deterministic and self.id==3: _remote_method(ParameterServer.write_evaluation, self.p_rref, ep+1, ep_reward)
            
        # if self.id==3:
        #     torch.save(self.local_policy.state_dict(), f"./model/{self.config.model_name}/actor_worker1.pth")
