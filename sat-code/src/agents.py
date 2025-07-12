import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

# Check if CUDA is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

def get_device_info():
    """
    Get information about the current device being used.
    
    Returns:
        dict: Device information including type, name, and memory stats
    """
    info = {
        'device': str(device),
        'cuda_available': torch.cuda.is_available()
    }
    
    if torch.cuda.is_available():
        info['device_name'] = torch.cuda.get_device_name(0)
        info['total_memory'] = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        info['allocated_memory'] = torch.cuda.memory_allocated(0) / 1024**3  # GB
        info['cached_memory'] = torch.cuda.memory_reserved(0) / 1024**3  # GB
        info['free_memory'] = info['total_memory'] - info['allocated_memory']
    
    return info

def print_gpu_memory_usage():
    """Print current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"GPU Memory - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB, Total: {total:.2f}GB")

class Actor(nn.Module):
    """
    Actor (Policy) Network for MADDPG.
    Each actor takes only its own state and outputs its own action.
    """
    def __init__(self, state_size, action_size, hidden_size=64):
        """
        Initialize parameters and build model.
        
        Args:
            state_size: Dimension of each state for this agent
            action_size: Dimension of each action for this agent
            hidden_size: Number of nodes in hidden layers
        """
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)
        
    def forward(self, state):
        """
        Build a network that maps state -> action values.
        
        Args:
            state: Agent state, shape (batch_size, state_size)
            
        Returns:
            Action values, shape (batch_size, action_size)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))  # Output in range [-1, 1]

class Critic(nn.Module):
    """
    Critic (Value) Network for MADDPG.
    The critic takes states and actions from all agents as input.
    """
    def __init__(self, state_size, action_size, num_agents, hidden_size=64):
        """
        Initialize parameters and build model.
        
        Args:
            state_size: Dimension of each state for a single agent
            action_size: Dimension of each action for a single agent
            num_agents: Number of agents in the environment
            hidden_size: Number of nodes in hidden layers
        """
        super(Critic, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        
        # First hidden layer processes concatenated states and actions from all agents
        self.fc1 = nn.Linear(state_size * num_agents + action_size * num_agents, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        
    def forward(self, states, actions):
        """
        Build a network that maps (state, action) pairs -> Q-values.
        
        Args:
            states: States of all agents, shape (batch_size, num_agents * state_size)
            actions: Actions of all agents, shape (batch_size, num_agents * action_size)
            
        Returns:
            Q-values, shape (batch_size, 1)
        """
        # Debug shapes
        batch_size = states.shape[0]
        expected_states_shape = (batch_size, self.state_size * self.num_agents)
        expected_actions_shape = (batch_size, self.action_size * self.num_agents)
        
        assert states.shape == expected_states_shape, f"Expected states shape {expected_states_shape}, got {states.shape}"
        assert actions.shape == expected_actions_shape, f"Expected actions shape {expected_actions_shape}, got {actions.shape}"
        
        # Concatenate states and actions
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class MADDPGAgent:
    def __init__(self, state_size, action_size, num_agents, agent_id, gamma=0.99, tau=0.01):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.agent_id = agent_id
        self.gamma = gamma  # Discount factor
        self.tau = tau      # Soft update parameter
        self.device = device  # Use the global device
        
        # Actor Networks - Each actor takes only its own state and outputs its own action
        self.actor = Actor(state_size, action_size).to(self.device)
        self.actor_target = Actor(state_size, action_size).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        
        # Critic Networks - Each critic takes the state and actions of all agents
        self.critic = Critic(state_size, action_size, num_agents).to(self.device)
        self.critic_target = Critic(state_size, action_size, num_agents).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-3)
        
        # Copy weights
        self.soft_update(self.actor_target, self.actor, 1.0)
        self.soft_update(self.critic_target, self.critic, 1.0)

    def update(self, experiences, next_actions, agent_id):
        """
        Update actor and critic networks.
        
        Args:
            experiences: Tuple of (states, actions, rewards, next_states, dones)
            next_actions: Tensor of next actions from all agents (batch_size, num_agents * action_size)
            agent_id: Index of this agent
        """
        states, actions, rewards, next_states, dones = experiences
        
        # Move tensors to device if they're not already
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        next_actions = next_actions.to(self.device)
        
        # Agent-specific reward and done
        agent_rewards = rewards[:, agent_id].unsqueeze(-1)
        agent_dones = dones[:, agent_id].unsqueeze(-1)
    
        # Update critic
        self.critic_optimizer.zero_grad()
        
        # Compute Q targets
        with torch.no_grad():
            Q_targets_next = self.critic_target(next_states, next_actions)
            Q_targets = agent_rewards + (self.gamma * Q_targets_next * (1 - agent_dones))
        
        # Get expected Q values
        Q_expected = self.critic(states, actions)
        
        # Compute critic loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        self.actor_optimizer.zero_grad()
        
        # Get the agent's state
        agent_state_idx = slice(agent_id * self.state_size, (agent_id + 1) * self.state_size)
        agent_state = states[:, agent_state_idx]
        
        # Get this agent's action based on its state
        agent_action = self.actor(agent_state)
        
        # Create a copy of the actions tensor and replace this agent's action
        actions_for_critic = actions.clone()
        agent_action_idx = slice(agent_id * self.action_size, (agent_id + 1) * self.action_size)
        actions_for_critic[:, agent_action_idx] = agent_action
        
        # Compute actor loss (negative of expected Q value)
        actor_loss = -self.critic(states, actions_for_critic).mean()
        
        # Minimize the loss
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
        """
        Returns actions for given state as per current policy.
        
        Args:
            state: Current state (can be numpy array or tensor)
            add_noise: Whether to add noise for exploration
            noise_scale: Scale of the noise
            
        Returns:
            Action values, numpy array
        """
        # Convert state to tensor if it's not already
        if not isinstance(state, torch.Tensor):
            state = torch.from_numpy(state).float()
        
        # Move to device
        state = state.to(self.device)
        
        # Make sure state has batch dimension
        if state.dim() == 1:
            state = state.unsqueeze(0)
        
        # Get action from policy network
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()  # Move back to CPU for numpy conversion
        self.actor.train()
        
        # Add noise for exploration
        if add_noise:
            noise = noise_scale * np.random.standard_normal(size=action.shape)
            action += noise
            action = np.clip(action, -1.0, 1.0)
            
        # Remove batch dimension if input was 1D
        if len(action) == 1:
            action = action.squeeze(0)
            
        return action

class ReplayBuffer:
    """
    Fixed-size buffer to store experience tuples.
    """
    def __init__(self, buffer_size, batch_size):
        """
        Initialize a ReplayBuffer object.
        
        Args:
            buffer_size: Maximum size of buffer
            batch_size: Size of each training batch
        """
        self.memory = []
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.position = 0
        
    def add(self, state, action, reward, next_state, done):
        """
        Add a new experience to memory.
        
        Args:
            state: Current state, numpy array
            action: Action taken, numpy array
            reward: Reward received, numpy array or scalar
            next_state: Next state, numpy array
            done: Done flag, numpy array or scalar
        """
        # Make sure all inputs are numpy arrays
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        if not isinstance(action, np.ndarray):
            action = np.array(action)
        if not isinstance(next_state, np.ndarray):
            next_state = np.array(next_state)
            
        # For reward and done, handle both scalar and vector cases
        if not isinstance(reward, np.ndarray):
            reward = np.array(reward)
        if not isinstance(done, np.ndarray):
            done = np.array(done)
            
        # Ensure state and next_state have consistent shapes
        if state.ndim == 1:
            state = state.reshape(1, -1)
        if action.ndim == 1:
            action = action.reshape(1, -1)
        if next_state.ndim == 1:
            next_state = next_state.reshape(1, -1)
        if reward.ndim == 1:
            reward = reward.reshape(1, -1)
        if done.ndim == 1:
            done = done.reshape(1, -1)
        
        # Store experience
        if len(self.memory) < self.buffer_size:
            self.memory.append((state, action, reward, next_state, done))
        else:
            # Replace oldest memory
            self.memory[self.position] = (state, action, reward, next_state, done)
            self.position = (self.position + 1) % self.buffer_size
        
    def sample(self):
        """
        Randomly sample a batch of experiences from memory.
        
        Returns:
            experiences: Tuple of (states, actions, rewards, next_states, dones)
        """
        batch_size = min(self.batch_size, len(self.memory))
        indices = np.random.choice(len(self.memory), batch_size, replace=False)
        experiences = [self.memory[i] for i in indices]
        
        # Convert to tensors - handle each component carefully
        states = torch.from_numpy(np.vstack([e[0] for e in experiences])).float().to(device)
        
        # For actions, ensure proper shape (batch_size, num_agents * action_size)
        actions_data = []
        for e in experiences:
            action = e[1]  # Should be shape (1, num_agents * action_size) or (num_agents * action_size,)
            if action.ndim == 2:
                # If it's 2D with batch dimension, take the first (and should be only) row
                action = action[0]
            actions_data.append(action)
        
        # Stack to create (batch_size, num_agents * action_size)
        actions = torch.from_numpy(np.array(actions_data)).float().to(device)
        
        rewards = torch.from_numpy(np.vstack([e[2] for e in experiences])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e[3] for e in experiences])).float().to(device)
        dones = torch.from_numpy(np.vstack([e[4] for e in experiences]).astype(np.uint8)).float().to(device)
        
        # Print shapes for debugging (only when there's an issue)
        # print(f"ReplayBuffer sample - states shape: {states.shape}, actions shape: {actions.shape}")
        # print(f"ReplayBuffer sample - rewards shape: {rewards.shape}, next_states shape: {next_states.shape}")
        # print(f"ReplayBuffer sample - dones shape: {dones.shape}")
        
        return (states, actions, rewards, next_states, dones)
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

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
        self.device = device  # Use the global device
        
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
        states_tensor = None
        
        # Convert states to tensor if it's a numpy array
        if isinstance(states, np.ndarray):
            states_tensor = torch.from_numpy(states).float().to(self.device)
        else:
            states_tensor = states.to(self.device)
            
        # Make sure it's the right shape
        if states_tensor.dim() == 1:
            # It's a flattened vector, so we'll split it for each agent
            for i, agent in enumerate(self.agents):
                # Extract state for this agent
                start_idx = i * self.state_size
                end_idx = (i + 1) * self.state_size
                agent_state = states_tensor[start_idx:end_idx].unsqueeze(0)  # Add batch dimension
                
                # Get action from agent
                action = agent.act(agent_state, add_noise, noise_scale)
                actions.append(action)
        else:
            # It's already batched, so we'll extract each agent's state
            batch_size = states_tensor.shape[0]
            for i, agent in enumerate(self.agents):
                # Extract state for this agent
                agent_idx = slice(i*self.state_size, (i+1)*self.state_size)
                agent_state = states_tensor[:, agent_idx]
                
                # Get action from agent
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
        # Make sure our inputs are properly shaped numpy arrays
        if isinstance(states, torch.Tensor):
            states = states.cpu().numpy()
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu().numpy()
        if isinstance(rewards, torch.Tensor):
            rewards = rewards.cpu().numpy()
        if isinstance(next_states, torch.Tensor):
            next_states = next_states.cpu().numpy()
        if isinstance(dones, torch.Tensor):
            dones = dones.cpu().numpy()
            
        # Convert 1D arrays to 2D arrays with batch dimension = 1
        if states.ndim == 1:
            states = states.reshape(1, -1)
        if next_states.ndim == 1:
            next_states = next_states.reshape(1, -1)
            
        # For actions, we need to flatten them properly for multi-agent case
        if actions.ndim == 2 and actions.shape == (self.num_agents, self.action_size):
            # Actions are in shape (num_agents, action_size), flatten to (1, num_agents * action_size)
            actions = actions.flatten().reshape(1, -1)
        elif actions.ndim == 1:
            # Already flattened, just add batch dimension
            actions = actions.reshape(1, -1)
            
        # For rewards and dones, make sure they're properly shaped
        if np.isscalar(rewards) or rewards.ndim == 0:
            rewards = np.array([rewards] * self.num_agents).reshape(1, -1)
        elif rewards.ndim == 1:
            rewards = rewards.reshape(1, -1)
            
        if np.isscalar(dones) or dones.ndim == 0:
            dones = np.array([dones] * self.num_agents).reshape(1, -1)
        elif dones.ndim == 1:
            dones = dones.reshape(1, -1)
        
        # Debug print to check shapes before storing (only when needed)
        # print(f"MADDPG step - storing: states {states.shape}, actions {actions.shape}, rewards {rewards.shape}")
        
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
        
        # Get the batch size
        batch_size = states.shape[0]
        
        # Create a tensor to hold next actions from all agents
        next_actions = torch.zeros(batch_size, self.num_agents * self.action_size, device=self.device)
        
        # Get next actions for each agent
        for i, agent in enumerate(self.agents):
            # Extract this agent's next state
            start_idx = i * self.state_size
            end_idx = (i + 1) * self.state_size
            agent_next_state = next_states[:, start_idx:end_idx]
            
            # Get next action from target network
            agent.actor_target.eval()
            with torch.no_grad():
                agent_next_action = agent.actor_target(agent_next_state)
            
            # Store in the appropriate part of next_actions tensor
            action_start_idx = i * self.action_size
            action_end_idx = (i + 1) * self.action_size
            next_actions[:, action_start_idx:action_end_idx] = agent_next_action
        
        # Update each agent
        for i, agent in enumerate(self.agents):
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
            agent.actor.load_state_dict(torch.load(f"{directory}/actor_agent_{i}.pth", map_location=self.device))
            agent.critic.load_state_dict(torch.load(f"{directory}/critic_agent_{i}.pth", map_location=self.device))
            agent.actor_target.load_state_dict(torch.load(f"{directory}/actor_agent_{i}.pth", map_location=self.device))
            agent.critic_target.load_state_dict(torch.load(f"{directory}/critic_agent_{i}.pth", map_location=self.device))