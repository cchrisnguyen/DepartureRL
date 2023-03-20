import torch.distributed.rpc as rpc
import numpy as np
import torch
import time
from datetime import datetime

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

    def __init__(self, env, p_rref, b_rref, shape, seed=0):
        self.id = rpc.get_worker_info().id
        self.local_policy = Policy(env.state_dim, env.action_dim, hid_shape=shape, seed=seed)
        self.env = env()
        self.p_rref = p_rref
        self.b_rref = b_rref
        self.env.seed(seed)


    def rollout(self, config):
        """Collect experience using an exploration policy"""
        for ep in range(config.n_episodes):
            # time0 = datetime.now()
            deterministic = True if (ep+1) % config.eval_interval == 0 else False
            while True:
                values = _remote_method(ParameterServer.pull, self.p_rref)
                if values is not None:
                    break
                time.sleep(0.5)

            self.local_policy.load_state_dict(values)
            state, ep_reward, done = self.env.reset(), 0, False
            operational_cost, noise_cost = 0, 0
            # states = []

            while not done:
                # Draw action from the distribution given by the actor
                if ep >= config.random_exploration_episodes:
                    state = torch.from_numpy(state).float()
                    action = self.local_policy.select_action(state, deterministic=deterministic)
                else:
                    action = np.random.uniform(-0.999, 1, self.env.action_dim)

                # Get environment's reponse
                # time1 = datetime.now()
                next_state, reward, done, dead, info= self.env.step(action)
                # time2 = datetime.now()
                ep_reward += reward
                operational_cost += info[1]
                noise_cost += info[2]

                _remote_method(PrioritizedReplay.store, self.b_rref, state, action, reward, next_state, dead)
                # time3 = datetime.now()
                # _remote_method(PrioritizedReplay.foo, self.b_rref)
                # time4 = datetime.now()
                state = next_state

            # End one episode
            print(f'Worker: {self.id-2:2d}\tEpisode: {ep+1:5d}\tReward:\t{ep_reward:8.2f}\tFinal position: {state[0] + self.env.initial_lat - self.env.target_lat:7.4f} {state[1]+ self.env.initial_lon - self.env.target_lon:7.4f} {state[2]:7.4f} {state[6]:7.4f}\t\tInfo: {info[0]}', flush=True)
            # print(f"Done in {datetime.now()-time0}, {time2 -time1}, {time3 -time2}, {time4 -time3}")
            _remote_method(ParameterServer.write, self.p_rref, self.id-3, ep+1, ep_reward, operational_cost, noise_cost)
            if deterministic and self.id==3: _remote_method(ParameterServer.write_evaluation, self.p_rref, ep+1, ep_reward)
            