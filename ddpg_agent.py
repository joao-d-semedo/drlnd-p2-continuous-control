import numpy as np
import random
from collections import namedtuple, deque

from network import MLP
from random_process import OrnsteinUhlenbeckProcess

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
ACTOR_LR = 1e-4         # actor network learning rate
CRITIC_LR = 1e-3        # critic network learning rate
UPDATE_EVERY = 1        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Actor
        self.actor_local         = MLP( sizes=[state_size, 400, 300, action_size], output_activation=nn.Tanh, seed=seed ).to(device)
        self.actor_target        = MLP( sizes=[state_size, 400, 300, action_size], output_activation=nn.Tanh, seed=seed ).to(device)
        self.actor_target.eval()
        self.actor_optimizer     = optim.Adam(self.actor_local.parameters(), lr=ACTOR_LR)
        self.actor_lr_scheduler  = optim.lr_scheduler.StepLR(self.actor_optimizer, step_size=400, gamma=.1)
        
        # Critic
        self.critic_local        = MLP( sizes=[state_size + action_size, 400, 300, 1], seed=seed ).to(device)
        self.critic_target       = MLP( sizes=[state_size + action_size, 400, 300, 1], seed=seed ).to(device)
        self.critic_target.eval()
        self.critic_optimizer    = optim.Adam(self.critic_local.parameters(), lr=CRITIC_LR)
        self.critic_lr_scheduler = optim.lr_scheduler.StepLR(self.critic_optimizer, step_size=400, gamma=.1)

        self.exploration_noise = OrnsteinUhlenbeckProcess(size=[action_size], std=0.2)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, exploring=False):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
        """

        state = torch.from_numpy(state).float().to(device)

        if exploring:
            self.actor_local.eval()
            with torch.no_grad():
                action = self.actor_local(state) + torch.from_numpy(self.exploration_noise.sample()).float().to(device)
            self.actor_local.train()
            return np.clip(action, -1, 1)
        else:
            return self.actor_local(state)


    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences
        n_exp = states.shape[0]

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            next_Q = self.critic_target(torch.cat( (next_states, next_actions), dim=1 ))

        Q_targets = rewards + gamma*next_Q*(1 - dones)
        
        Q_predictions = self.critic_local(torch.cat( (states, actions), dim=1 ))
        
        critic_loss = F.mse_loss(Q_predictions, Q_targets)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -self.critic_local(torch.cat( (states, self.actor_local(states)), dim=1 )).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.actor_local,  self.actor_target,  TAU)
        self.soft_update(self.critic_local, self.critic_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)