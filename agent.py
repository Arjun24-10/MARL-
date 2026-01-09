import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque
from model import QNetwork

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # 1. Two Networks for stability
        self.policy_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        
        # 2. Experience Replay Memory
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.gamma = 0.95
        self.epsilon = 1.0 
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        # Epsilon-greedy exploration
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_dim)
        
        state_t = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_net(state_t)
        return torch.argmax(q_values).item()

    def replay(self):
        # 1. Guard clause: Ensure we have enough data to train
        if len(self.memory) < self.batch_size:
            return

        # 2. Sample a random batch
        minibatch = random.sample(self.memory, self.batch_size)
        
        # 3. Vectorize data into Tensors (FAST)
        states = torch.FloatTensor(np.array([m[0] for m in minibatch]))
        actions = torch.LongTensor([m[1] for m in minibatch])
        rewards = torch.FloatTensor([m[2] for m in minibatch])
        next_states = torch.FloatTensor(np.array([m[3] for m in minibatch]))
        dones = torch.FloatTensor([m[4] for m in minibatch])

        # 4. Get current Q-values from Policy Network
        # We gather the Q-values for the specific actions taken
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1))

        # 5. Get maximum Q-values for next states from Target Network
        with torch.no_grad():
            max_next_q = self.target_net(next_states).max(1)[0]
            # Bellman Equation
            target_q = rewards + (1 - dones) * self.gamma * max_next_q

        # 6. Compute Loss and Optimize
        loss = nn.MSELoss()(current_q.squeeze(), target_q)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target(self):
        # Sync target network with policy network
        self.target_net.load_state_dict(self.policy_net.state_dict())