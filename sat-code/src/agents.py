import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class Actor(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

        # Actor Networks
        self.actor = Actor(state_size, action_size)
        self.actor_target = Actor(state_size, action_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        
        # Critic Networks
        self.critic = Critic(state_size, action_size, num_agents)
        self.critic_target = Critic(state_size, action_size, num_agents)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Copy weights
        self.soft_update(self.actor_target, self.actor, 1.0)
        self.soft_update(self.critic_target, self.critic, 1.0)

class Critic(nn.Module):
    def __init__(self, state_size, action_size, num_agents, hidden_size=64):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size * num_agents + action_size * num_agents, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class MADDPGAgent:
    def __init__(self, state_size, action_size, num_agents, agent_id):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        
        # Actor Networks
        self.actor = Actor(state_size, action_size)
        self.actor_target = Actor(state_size, action_size)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        
        # Critic Networks
        self.critic = Critic(state_size, action_size, num_agents)
        self.critic_target = Critic(state_size, action_size, num_agents)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Copy weights
        self.soft_update(self.actor_target, self.actor, 1.0)
        self.soft_update(self.critic_target, self.critic, 1.0)

    def update(self, experiences, next_actions, agent_id):
        states, actions, rewards, next_states, dones = experiences
    
        # Update critic
        Q_targets_next = self.critic_target(next_states, next_actions)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        Q_expected = self.critic(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update target networks
        self.soft_update(self.critic_target, self.critic, self.tau)
        self.soft_update(self.actor_target, self.actor, self.tau)
        

class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.memory = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        
    def add(self, state, action, reward, next_state, done):
        if len(self.memory) >= self.buffer_size:
            self.memory.pop(0)
        self.memory.append((state, action, reward, next_state, done))
        
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float()
        actions = torch.from_numpy(np.vstack([e[1] for e in experiences])).float()
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float()
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float()
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float()
        return (states, actions, rewards, next_states, dones)