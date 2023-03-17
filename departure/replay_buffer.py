import numpy as np
from collections import deque

class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=int(1e5)):
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 #for beta calculation
        self.capacity   = capacity
        self.buffer     = deque(maxlen=capacity)
        self.pos        = 0
        self.priorities = deque(maxlen=capacity)
        self.rollout_steps = 0
    
    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        3.4 ANNEALING THE BIAS (Paper: PER)
        We therefore exploit the flexibility of annealing the amount of importance-sampling
        correction over time, by defining a schedule on the exponent 
        that reaches 1 only at the end of
        learning. In practice, we linearly anneal 
        from its initial value 
        0 to 1
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)
    
    def store(self, states, actions, rewards, next_states, deads):
        n_samples = len(states)
        max_prio = max(self.priorities) if self.buffer else 1.0 # gives max priority if buffer is not empty else 1
        
        self.buffer.extendleft(zip(states, actions, rewards, next_states, deads))
        self.priorities.extendleft([max_prio]*n_samples)
        self.rollout_steps += 1
    
    def sample(self, batch_size, c_k):
        N = len(self.buffer)
        if N < batch_size: return None
        
        if c_k > N: c_k = N
        prios = np.array(self.priorities)[:c_k]

        # calc P = p^a/sum(p^a)
        probs  = prios ** self.alpha
        P = probs/probs.sum()
        
        #gets the indices depending on the probability p and the c_k range of the buffer
        indices = np.random.choice(c_k, batch_size, p=P) 
        samples = [self.buffer[idx] for idx in indices]
        
        beta = self.beta_by_frame(self.frame)
        self.frame+=1
                
        #Compute importance-sampling weight
        weights  = (c_k * P[indices]) ** (-beta)
        # normalize weights
        weights /= weights.max() 
        weights  = np.array(weights, dtype=np.float32) 
        
        states, actions, rewards, next_states, dones = zip(*samples) 
        return np.vstack(states), np.vstack(actions), rewards, np.vstack(next_states), dones, indices, weights
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = abs(prio)

    def __len__(self):
        return len(self.buffer)
    
    def get_counts(self):
        return self.rollout_steps
