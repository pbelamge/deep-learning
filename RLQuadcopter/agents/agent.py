from agents.actor import Actor
from agents.critic import Critic
from agents.utils import *

import numpy as np

class DDPG ():
    """ Reinforcement Learning Agent. """
    def __init__ (self, task, exp_mu, exp_theta, exp_sigma, gamma, tau):
        self.task = task
        
        self.s_size = task.s_size
        self.a_size = task.a_size
        
        self.a_low = task.a_low
        self.a_high = task.a_high
        
        # Actor Model
        self.actor_local = Actor (self.s_size, self.a_size, self.a_low, self.a_high)
        self.actor_target = Actor (self.s_size, self.a_size, self.a_low, self.a_high)
        
        # Critic Model
        self.critic_local = Critic (self.s_size, self.a_size)
        self.critic_target = Critic (self.s_size, self.a_size)
        
        # Initialize target model parameters
        self.critic_target.model.set_weights (self.critic_local.model.get_weights ())
        self.actor_target.model.set_weights (self.actor_local.model.get_weights ())
        
        # initialize noise
        self.exp_mu = exp_mu
        self.exp_theta = exp_theta
        self.exp_sigma = exp_sigma
        self.noise = OUNoise (self.a_size, self.exp_mu, self.exp_theta, self.exp_sigma)
        
        # For Replay buffer
        self.buff_size = 1024 * 1024
        self.batch_size = 64
        self.memory = ReplayBuffer (self.buff_size, self.batch_size)
        
        # discount factor
        self.gamma = gamma
        
        # for soft update of target parameters
        self.tau = tau
        
    def reset_episode (self):
        self.noise.reset ()
        state = self.task.reset ()
        
        # last state
        self.l_state = state
        
        return state
    
    # A - Action, R - Reward, D - Done
    def step (self, A, R, nState, D):
        # save experience to memory
        self.memory.add (self.l_state, A, R, nState, D)
        
        # Learn, if enough samples (experiences) are available in memory
        if len (self.memory) > self.batch_size:
            self.learn (self.memory.sample ())
        
        self.l_state = nState
        
    def act (self, states):
        S = np.reshape (states, [-1, self.s_size])
        A = self.actor_local.model.predict (S)[0]
        return list (A + self.noise.sample ())
    
    def learn (self, exp):
        """Update policy and value parameters using given batch of experience tuples."""

        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        S = np.vstack([e.state for e in exp if e is not None])
        A = np.array([e.action for e in exp if e is not None]).astype(np.float32).reshape(-1, self.a_size)
        R = np.array([e.reward for e in exp if e is not None]).astype(np.float32).reshape(-1, 1)
        D = np.array([e.done for e in exp if e is not None]).astype(np.uint8).reshape(-1, 1)
        nS = np.vstack([e.next_state for e in exp if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        a_next = self.actor_target.model.predict_on_batch(nS)
        t_next = self.critic_target.model.predict_on_batch([nS, a_next])

        # Compute Q targets for current state and train critic model (local)
        Q_targets = R + self.gamma * t_next * (1 - D)
        self.critic_local.model.train_on_batch(x=[S, A], y=Q_targets)

        # Train actor model (local)
        a_grad = np.reshape(self.critic_local.get_action_gradients([S, A, 0]), (-1, self.a_size))
        self.actor_local.train_fn([S, a_grad, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        l_weights = np.array(local_model.get_weights())
        t_weights = np.array(target_model.get_weights())

        assert len(l_weights) == len(t_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * l_weights + (1 - self.tau) * t_weights
        target_model.set_weights(new_weights)