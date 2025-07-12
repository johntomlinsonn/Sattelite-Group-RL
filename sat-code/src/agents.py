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
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Output in range [-1, 1]

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
    def __init__(self, state_size, action_size, num_agents, agent_id, gamma=0.99, tau=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.agent_id = agent_id
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update parameter
        
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
        
    def soft_update(self, target, source, tau):
        """
        Soft update model parameters: θ_target = τ*θ_source + (1 - τ)*θ_target
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - tau) * target_param.data + tau * source_param.data)

    def act(self, state, add_noise=True, noise_scale=0.1):
        """Get action from actor network, optionally with noise for exploration"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).squeeze().numpy()
        self.actor.train()
        
        if add_noise:
            action += noise_scale * np.random.standard_normal(action.shape)
            action = np.clip(action, -1, 1)
            
        return action

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

class MADDPG:
    """
    MADDPG Trainer class that manages multiple agents and coordinates their training.
    """
    def __init__(self, state_size, action_size, num_agents, buffer_size=100000, batch_size=128, gamma=0.99, tau=0.01):
        """
        Initialize a MADDPG trainer.
        
        Args:
            state_size: Size of the state space for each agent
            action_size: Size of the action space for each agent
            num_agents: Number of agents in the environment
            buffer_size: Size of the replay buffer
            batch_size: Batch size for training
            gamma: Discount factor
            tau: Soft update parameter
        """
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        
        # Create multiple agents
        self.agents = [MADDPGAgent(state_size, action_size, num_agents, i, gamma, tau) 
                      for i in range(num_agents)]
        
        # Shared replay buffer
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size)
        
    def act(self, states, add_noise=True, noise_scale=0.1):
        """
        Get actions from all agents based on current states.
        
        Args:
            states: Array of states for all agents (flattened)
            add_noise: Whether to add noise for exploration
            noise_scale: Scale of the noise for exploration
            
        Returns:
            actions: Array of actions for all agents
        """
        actions = []
        for i, agent in enumerate(self.agents):
            # Extract state for this agent
            agent_state = states[i*self.state_size:(i+1)*self.state_size]
            action = agent.act(agent_state, add_noise, noise_scale)
            actions.append(action)
            
        return np.array(actions)
    
    def step(self, states, actions, rewards, next_states, dones):
        """
        Save experience in replay buffer and perform learning if enough samples are available.
        
        Args:
            states: Array of current states for all agents
            actions: Array of actions taken by all agents
            rewards: Array of rewards received by all agents
            next_states: Array of next states for all agents
            dones: Array of done flags for all agents
        """
        # Save experience in replay buffer
        self.replay_buffer.add(states, actions, rewards, next_states, dones)
        
        # Learn if enough samples are available in memory
        if len(self.replay_buffer.memory) > self.batch_size:
            experiences = self.replay_buffer.sample()
            self.learn(experiences)
    
    def learn(self, experiences):
        """
        Update policy and value parameters using given batch of experience tuples.
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones)
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Get predicted next-state actions and Q values from target models
        next_actions = torch.zeros((states.shape[0], self.action_size * self.num_agents), device=states.device)
        
        # Get next actions for each agent
        for i, agent in enumerate(self.agents):
            agent_idx = torch.arange(i*self.state_size, (i+1)*self.state_size)
            agent_next_state = next_states[:, agent_idx]
            agent_next_action = agent.actor_target(agent_next_state)
            next_actions[:, i*self.action_size:(i+1)*self.action_size] = agent_next_action
        
        # Update each agent
        for i, agent in enumerate(self.agents):
            # Get reward for this agent
            agent_reward = rewards[:, i].unsqueeze(-1)
            agent_done = dones[:, i].unsqueeze(-1)
            
            # Update agent
            agent.update(experiences, next_actions, i)
    
    def save(self, directory):
        """
        Save trained models for all agents.
        
        Args:
            directory: Directory to save the models
        """
        import os
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        for i, agent in enumerate(self.agents):
            torch.save(agent.actor.state_dict(), f"{directory}/actor_agent_{i}.pth")
            torch.save(agent.critic.state_dict(), f"{directory}/critic_agent_{i}.pth")
            
    def load(self, directory):
        """
        Load trained models for all agents.
        
        Args:
            directory: Directory to load the models from
        """
        for i, agent in enumerate(self.agents):
            agent.actor.load_state_dict(torch.load(f"{directory}/actor_agent_{i}.pth"))
            agent.critic.load_state_dict(torch.load(f"{directory}/critic_agent_{i}.pth"))
            agent.actor_target.load_state_dict(torch.load(f"{directory}/actor_agent_{i}.pth"))
            agent.critic_target.load_state_dict(torch.load(f"{directory}/critic_agent_{i}.pth"))