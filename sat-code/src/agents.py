import torch
import torch.nn as nn
import torch.nn.functional as F

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