import argparse
import os
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from torch.distributed.rpc import rpc_async, remote

from departure_env import Departure
from sac import SAC
from misc import _call_method, _remote_method
from worker import Worker
from replay_buffer import PrioritizedReplay
from parameter_server import ParameterServer

def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True','true','TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False','false','FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def run(rank, config):
    r"""
    This is the entry point for all processes. 
    The rank 0 is the master agent. 
    All other ranks are workers.
    """

    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(config.master_port)

    if rank == 0:
        # rank0 is the master agent
        rpc.init_rpc("learner", rank=rank, world_size=config.n_rollout_threads+3)
        
        buffer_ref = remote(rpc.get_worker_info('buffer'), PrioritizedReplay, args=(config.buffer_length,), kwargs={'beta_frames':int(1e6)})
        ps_ref = remote(rpc.get_worker_info('ps'), ParameterServer, args=(config,))

        w_rrefs  = []
        for w_rank in range(config.n_rollout_threads):
            w_info = rpc.get_worker_info(str(w_rank + 3))
            w_rrefs.append(remote(w_info, Worker, args=(Departure, ps_ref, buffer_ref, config.hid_shape)))

        l_rref = remote(rpc.get_worker_info('learner'), SAC,
                            args=(Departure,
                            buffer_ref,
                            ps_ref,
                            config)
                        )

        futs = []
        

        for w_rref in w_rrefs:
            # make async RPC to kick off n episodes on all observers
            futs.append(
                rpc_async(
                    w_rref.owner(),
                    _call_method,
                    args=(Worker.rollout, w_rref, config),
                    timeout=int(0)
                )
            )

        futs.append(rpc_async(l_rref.owner(), 
                    _call_method,
                    args=(SAC.learn, l_rref),
                    timeout=int(0)))
        
        # # wait until all obervers have finished this episode
        [fut.wait() for fut in futs]

        _remote_method(SAC.save, l_rref, 0)
        
    elif rank == 1:
        rpc.init_rpc('buffer', rank=rank, world_size=config.n_rollout_threads+3)
    elif rank == 2:
        rpc.init_rpc('ps', rank=rank, world_size=config.n_rollout_threads+3)
    else:
        # other ranks are workers
        rpc.init_rpc(str(rank), rank=rank, world_size=config.n_rollout_threads+3)
    
    rpc.shutdown()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name",
                        help="Name of directory to store " +
                             "model/training contents")
    
    parser.add_argument("--master_port", default=29500, type=int)

    parser.add_argument('--write', type=str2bool, default=True, help='Use SummaryWriter to record the training')
    parser.add_argument('--render', type=str2bool, default=False, help='Render or Not')
    parser.add_argument('--model_index', type=int, default=5, help='which epoch to load')

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--n_rollout_threads", default=4, type=int)
    parser.add_argument("--buffer_length", default=int(1e6), type=int)
    parser.add_argument("--n_episodes", default=10000, type=int)
    parser.add_argument('--random_exploration_episodes', default=200, type=int)
    parser.add_argument("--steps_per_update", default=100, type=int)
    
    parser.add_argument('--update_freq', default=1, type=int,
                        help='the number of updates until the next push')
    parser.add_argument('--start_learning_after', default=10000, type=int,
                        help='the number of experience records in replay buffer that the agent start learning')
    parser.add_argument('--learn_after_rollout', default=1000, type=int,
                        help='the number of updates after finishing all roll-out tasks')
    parser.add_argument("--batch_size", default=1024, type=int,
                        help="Batch size for training")
    
    parser.add_argument('--save_interval', type=int, default=50, help='Model saving interval, in epochs.')
    parser.add_argument('--eval_interval', type=int, default=20, help='Model evaluating interval, in epochs.')

    parser.add_argument('--hid_shape', nargs='+', default = [256, 128, 128, 64], type=int)
    parser.add_argument("--alpha", default=0.2, type=float)
    parser.add_argument("--adaptive_alpha", default=False, type=bool)
    parser.add_argument("--pi_lr", default=1e-5, type=float)
    parser.add_argument("--q_lr", default=1e-4, type=float)
    parser.add_argument("--tau", default=0.001, type=float)
    parser.add_argument("--gamma", default=0.99, type=float)
    parser.add_argument("--device", default='cuda:0', type=str)

    config = parser.parse_args()
    print(config)
    
    if config.render:
        agent = SAC.init_for_testing(Departure, config)
        reward = agent.evaluate(Departure, write_csv=True)
        print('Average Reward:', reward)
    else:
        path = f"./model/{config.model_name}"
        if not os.path.exists(path): os.makedirs(path)
        mp.spawn(
            run,
            args=(config, ),
            nprocs=config.n_rollout_threads+3,
            join=True
        )
