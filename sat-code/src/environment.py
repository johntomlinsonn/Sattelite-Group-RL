import gymnasium as gym
import numpy as np
import random as rand
import pygame
import sys

class SatelliteSwarmEnv(gym.Env):
    def __init__(self, num_satellites, grid_size, max_timesteps, coverage_radius, comm_range=15):
        super(SatelliteSwarmEnv, self).__init__()
        # Satellite settings
        self.num_satellites = num_satellites
        self.coverage_radius = coverage_radius
        self.comm_range = comm_range 

        # Environment state
        self.current_step = 0
        self.max_timesteps = max_timesteps
    
        # Grid settings
        self.grid_width = grid_size
        self.grid_height = grid_size
        self.earth_coverage_map = np.zeros((self.grid_width, self.grid_height), dtype=bool)

        # Initialize satellites
        self.satellites = [Sat(self.grid_width, self.grid_height) for _ in range(num_satellites)]
        
        # Define action and observation spaces
        # Actions: [dx, dy] for each satellite (continuous)
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(num_satellites, 2),
            dtype=np.float32
        )

        # Observations: flattened array containing positions and velocities
        # Format: [pos_x1, pos_y1, vel_x1, vel_y1, pos_x2, pos_y2, vel_x2, vel_y2, ...]
        self.observation_space = gym.spaces.Box(
            low=np.array([0, 0, -10, -10] * num_satellites),
            high=np.array([grid_size, grid_size, 10, 10] * num_satellites),
            shape=(num_satellites * 4,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Reset all satellites
        for sat in self.satellites:
            sat.reset()
        
        return self._get_observation(), {}

    def _get_observation(self):
        """Return a flattened array of all satellite positions and velocities."""
        observation = []
        for sat in self.satellites:
            x, y = sat.getPosition()
            vx, vy = sat.velocity
            observation.extend([x, y, vx, vy])
        return np.array(observation, dtype=np.float32)

    def calculate_coverage(self):
        """Calculate the percentage of Earth grid covered by satellites."""
        self.earth_coverage_map.fill(False)
        for sat in self.satellites:
            x, y = sat.getPosition()
            for i in range(max(0, int(x - self.coverage_radius)), min(self.grid_width, int(x + self.coverage_radius + 1))):
                for j in range(max(0, int(y - self.coverage_radius)), min(self.grid_height, int(y + self.coverage_radius + 1))):
                    if (i - x) ** 2 + (j - y) ** 2 <= self.coverage_radius ** 2:
                        self.earth_coverage_map[i, j] = True
        return np.sum(self.earth_coverage_map) / (self.grid_width * self.grid_height)

    def _calculate_reward(self):
        """Calculate the reward based on coverage and communication links."""
        coverage_reward = self.calculate_coverage()
        
        # Calculate boundary penalty - negative reward for satellites near edges
        boundary_penalty = 0.0
        boundary_margin = 5  # Distance from edge where penalty starts
        
        for sat in self.satellites:
            x, y = sat.getPosition()
            
            # Calculate distance to nearest boundary
            dist_to_left = x
            dist_to_right = self.grid_width - 1 - x
            dist_to_top = y
            dist_to_bottom = self.grid_height - 1 - y
            
            # Find minimum distance to any boundary
            min_dist_to_boundary = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
            
            # Apply penalty if too close to boundary
            if min_dist_to_boundary < boundary_margin:
                # Penalty increases as satellite gets closer to edge
                penalty_strength = (boundary_margin - min_dist_to_boundary) / boundary_margin
                boundary_penalty -= 0.1 * penalty_strength  # Adjust penalty strength as needed
        
        # Combine rewards with weights
        total_reward = coverage_reward + boundary_penalty
        return total_reward

    def _calculate_per_agent_rewards(self):
        """Calculate individual rewards for each satellite based on their performance."""
        rewards = []
        
        for i, sat in enumerate(self.satellites):
            # Individual coverage contribution
            individual_coverage = 0
            x, y = sat.getPosition()
            
            # Count coverage points this satellite is responsible for
            for gx in range(max(0, int(x - self.coverage_radius)), 
                          min(self.grid_width, int(x + self.coverage_radius + 1))):
                for gy in range(max(0, int(y - self.coverage_radius)), 
                              min(self.grid_height, int(y + self.coverage_radius + 1))):
                    distance = np.sqrt((gx - x)**2 + (gy - y)**2)
                    if distance <= self.coverage_radius:
                        individual_coverage += 1
            
            # Normalize by total possible coverage
            max_coverage = np.pi * self.coverage_radius**2
            coverage_reward = individual_coverage / max_coverage
            
            # Individual boundary penalty
            boundary_penalty = 0.0
            boundary_margin = 5
            
            dist_to_left = x
            dist_to_right = self.grid_width - 1 - x
            dist_to_top = y
            dist_to_bottom = self.grid_height - 1 - y
            min_dist_to_boundary = min(dist_to_left, dist_to_right, dist_to_top, dist_to_bottom)
            
            if min_dist_to_boundary < boundary_margin:
                penalty_strength = (boundary_margin - min_dist_to_boundary) / boundary_margin
                boundary_penalty = -0.1 * penalty_strength
            
            # Communication bonus (reward for being connected to other satellites)
            comm_bonus = 0.0
            for j, other_sat in enumerate(self.satellites):
                if i != j:
                    other_x, other_y = other_sat.getPosition()
                    distance = np.sqrt((x - other_x)**2 + (y - other_y)**2)
                    if distance <= self.comm_range:
                        comm_bonus += 0.05  # Small bonus for each connection
            
            # Combine individual rewards
            total_reward = coverage_reward + boundary_penalty + comm_bonus
            rewards.append(total_reward)
        
        return np.array(rewards)

    def step(self, action):
        self.current_step += 1
        
        # Apply actions (accelerations) to satellites
        for i, sat in enumerate(self.satellites):
            dx, dy = action[i]
            current_vx, current_vy = sat.velocity
            
            # Update velocity (with some damping to prevent extreme velocities)
            new_vx = 0.95 * current_vx + dx
            new_vy = 0.95 * current_vy + dy
            
            # Clip velocities
            new_vx = np.clip(new_vx, -10, 10)
            new_vy = np.clip(new_vy, -10, 10)
            
            sat.updateVelocity(new_vx, new_vy)
            
            # Update position
            x, y = sat.getPosition()
            new_x = np.clip(x + new_vx, 0, self.grid_width - 1)
            new_y = np.clip(y + new_vy, 0, self.grid_height - 1)
            sat.position = (new_x, new_y)
        
        # Calculate reward
        # Choose between global reward or per-agent rewards
        # For global reward (current behavior):
        reward = self._calculate_reward()
        
        # For per-agent rewards, uncomment the line below and comment out the line above:
        # reward = self._calculate_per_agent_rewards()
        
        # Check if episode is done
        done = self.current_step >= self.max_timesteps
        
        return self._get_observation(), reward, done, False, {}
    
class SatelliteRenderer:
    def __init__(self, env, width=800, height=800):
        self.env = env
        self.width = width
        self.height = height
        self.grid_width = env.grid_width
        self.grid_height = env.grid_height
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Satellite Coverage Visualization')
        self.clock = pygame.time.Clock()

    def draw(self):
        self.screen.fill((255, 255, 255))
        # Draw grid (optional)
        for i in range(0, self.width, self.width // self.grid_width):
            pygame.draw.line(self.screen, (220,220,220), (int(i),0), (int(i),self.height))
        for j in range(0, self.height, self.height // self.grid_height):
            pygame.draw.line(self.screen, (220,220,220), (0,int(j)), (self.width,int(j)))
        # Draw satellites and coverage
        for sat in self.env.satellites:
            x, y = sat.getPosition()
            sx = int(x * self.width / self.grid_width)
            sy = int(y * self.height / self.grid_height)
            coverage_px = int(self.env.coverage_radius * self.width / self.grid_width)
            pygame.draw.circle(self.screen, (0, 0, 255), (sx, sy), coverage_px, width=0)
            pygame.draw.circle(self.screen, (255, 0, 0), (sx, sy), 5)
        pygame.display.flip()

    def run(self, steps=500, fps=30):
        """
        Run the visualization for a specified number of steps with random actions.
        
        Args:
            steps: Number of steps to run
            fps: Frames per second for visualization
        """
        for _ in range(steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # Generate random actions for all satellites
            random_actions = np.random.uniform(-1, 1, size=(self.env.num_satellites, 2))
            self.env.step(random_actions)
            
            self.draw()
            self.clock.tick(fps)
    
class Sat:
    def __init__(self,grid_x_size, grid_y_size):
        self.position = (None, None)
        self.velocity = (None, None)
        self.grid_x_ = grid_x_size
        self.grid_y_ = grid_y_size
        self.reset()

    def reset(self):
        self.position = (rand.randint(0,self.grid_x_), rand.randint(0,self.grid_y_))
        self.velocity = (0, 0)

    def updateVelocity(self, velocity_x, velocity_y):
        self.velocity = (velocity_x, velocity_y)
    
    def randomizeVelocity(self):
        self.velocity = (rand.randint(-10, 10), rand.randint(-10, 10))

    def getPosition(self):
        return self.position

