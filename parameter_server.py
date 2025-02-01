import pickle
from collections import OrderedDict
import os, shutil
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter
from threading import Semaphore

class ParameterServer(object):
    def __init__(self, config):
        """
        Initialize the ParameterServer.
        
        Parameters:
        - config (object): Configuration object containing various settings
        """
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
        """
        Push new weights to the parameter server.
        
        Parameters:
        - weights (OrderedDict): The new weights to be pushed
        """
        semaphore_counts = 0
        while True:
            self.semaphore.acquire()
            semaphore_counts += 1
            if semaphore_counts == self.config.n_rollout_threads: break

        self.weights = weights
        self.semaphore.release(self.config.n_rollout_threads)

    def pull(self):
        """
        Pull the latest weights from the parameter server.
        
        Returns:
        - OrderedDict: The latest weights
        """
        self.semaphore.acquire()
        weights = self.weights
        self.semaphore.release()
        return weights

    def log(self, tag, value, step):
        """
        Log a value to TensorBoard.
        
        Parameters:
        - tag (str): The tag associated with the value
        - value (float): The value to log
        - step (int): The current step or epoch
        """
        if self.writer:
            self.writer.add_scalar(tag, value, step)

    def write(self, worker_id, ep, reward, operational_cost, noise_cost, nfz_penalty):
        if self.writer: 
            self.writer.add_scalar(f'Worker{worker_id}_Reward', reward, global_step=ep)
            self.writer.add_scalar(f'Worker{worker_id}_Operation', operational_cost, global_step=ep)
            self.writer.add_scalar(f'Worker{worker_id}_Noise', noise_cost, global_step=ep)
            self.writer.add_scalar(f'Worker{worker_id}_NFZ', nfz_penalty, global_step=ep)
        if worker_id==1 and ep % self.config.save_interval == 0: self.save(ep)

    def write_evaluation(self, ep, reward):
        if self.writer: self.writer.add_scalar('Evaluation', reward, global_step=ep)

    def save(self, ep):
        torch.save(self.weights, f"./model/{self.config.model_name}/actor_{ep}.pth")
