import pickle
from collections import OrderedDict
import os, shutil
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from threading import Semaphore

class ParameterServer(object):
    def __init__(self, config):
        self.config = config
        self.weights = None
        self.writer = None
        if config.write:
            timenow = datetime.now().strftime('%m-%dT%H:%M')
            writepath = f'runs/{config.model_name}_{timenow}'
            if os.path.exists(writepath): shutil.rmtree(writepath)
            self.writer = SummaryWriter(log_dir=writepath)
        
        self.semaphore = Semaphore(config.n_rollout_threads)
    
    def push(self, weights):
        # print("PUSH")
        semaphore_counts = 0
        while True:
            self.semaphore.acquire()
            semaphore_counts += 1
            if semaphore_counts == self.config.n_rollout_threads: break

        self.weights = weights # OrderedDict({layer: weights[layer].detach().clone() for layer in weights})
        self.semaphore.release(self.config.n_rollout_threads)
        # print("PUSH finished")

    def pull(self):
        # print("PULL")
        self.semaphore.acquire()
        weights = None
        if self.weights is not None:
            weights = OrderedDict({layer: self.weights[layer].detach().clone() for layer in self.weights})
        self.semaphore.release()
        # print("PULL finished")
        return weights


    def write(self, worker_id, ep, reward, operational_cost, noise_cost):
        if self.writer: 
            self.writer.add_scalar(f'Worker{worker_id}_Reward', reward, global_step=ep)
            self.writer.add_scalar(f'Worker{worker_id}_Operation', operational_cost, global_step=ep)
            self.writer.add_scalar(f'Worker{worker_id}_Noise', noise_cost, global_step=ep)
        if worker_id==1 and ep % self.config.save_interval == 0: self.save(ep)

    def write_evaluation(self, ep, reward):
        if self.writer: self.writer.add_scalar('Evaluation', reward, global_step=ep)

    def save(self, ep):
        torch.save(self.weights, f"./model/{self.config.model_name}/actor_{ep}.pth")
