import numpy as np
from collections import deque

class PrioritizedReplay(object):
    """
    Proportional Prioritization
    """
    def __init__(self, capacity, alpha=0.6, beta_start=0.4, beta_frames=int(1e5)):
        """
        Initialize the PrioritizedReplay buffer.
        
        Parameters:
        - capacity (int): Maximum number of elements in the buffer
        - alpha (float): How much prioritization is used (0 - no prioritization, 1 - full prioritization)
        - beta_start (float): Initial value of beta for importance sampling
        - beta_frames (int): Number of frames over which beta is annealed from beta_start to 1
        """
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.frame = 1 # for beta calculation
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.pos = 0
        self.priorities = deque(maxlen=capacity)
        self.rollout_steps = 0

    def beta_by_frame(self, frame_idx):
        """
        Linearly increases beta from beta_start to 1 over time from 1 to beta_frames.
        
        Parameters:
        - frame_idx (int): The current frame index
        
        Returns:
        - float: The value of beta for the given frame index
        """
        return min(1.0, self.beta_start + frame_idx * (1.0 - self.beta_start) / self.beta_frames)

    def store(self, state, action, reward, next_state, done):
        assert state.ndim == next_state.ndim
        state      = np.expand_dims(state, 0)
        action      = np.expand_dims(action, 0)
        next_state = np.expand_dims(next_state, 0)
        max_prio = max(self.priorities) if self.buffer else 1.0 # gives max priority if buffer is not empty else 1
        self.buffer.appendleft((state, action, reward, next_state, done))
        self.priorities.appendleft(max_prio)
        self.rollout_steps += 1

    def sample(self, batch_size, beta=None):
        """
        Sample a batch of experiences from the buffer.
        
        Parameters:
        - batch_size (int): The number of experiences to sample
        - beta (float): The value of beta for importance sampling
        
        Returns:
        - tuple: Sampled experiences and their corresponding indices and weights
        """
        if beta is None:
            beta = self.beta_by_frame(self.frame)
        self.frame += 1

        prios = np.array(self.priorities)
        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        """
        Update the priorities of sampled experiences.
        
        Parameters:
        - batch_indices (list): Indices of the sampled experiences
        - batch_priorities (list): New priorities for the sampled experiences
        """
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

    def get_counts(self):
        return self.rollout_steps