import numpy as np
from collections import namedtuple, deque
import random

class ReplayBuffer ():
    """Fixed-size buffer to store experience tuples."""
    
    def __init__ (self, buff_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buff_size: maximum size of buffer
            batch_size: size of each training batch
        """
        
        self.memory = deque (maxlen = buff_size)
        self.batch_size = batch_size
        self.experience = namedtuple ("Experience", field_names = ["state", "action", "reward", "next_state", "done"])
    
    def add (self, S, A, R, nS, D):
        """Add a new experience to memory.
        Params
        ======
            S: State, A: Action, R: Reward, nS: Next State, D: Done
        """
        
        e = self.experience (S, A, R, nS, D)
        self.memory.append (e)
    
    def sample (self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        
        return random.sample (self.memory, k = self.batch_size)
    
    def __len__ (self):
        return len (self.memory)
    
class OUNoise ():
    """Ornstein-Uhlenbeck process."""
    
    def __init__ (self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        
        self.mu = mu * np.ones (size)
        self.theta = theta
        self.sigma = sigma
        self.reset ()
    
    def reset (self):
        """Reset the internal state (= noise) to mean (mu)."""
        
        self.state = self.mu
    
    def sample (self):
        """Update internal state and return it as a noise sample."""
        
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn (len (self.state))
        self.state += dx
        
        return self.state