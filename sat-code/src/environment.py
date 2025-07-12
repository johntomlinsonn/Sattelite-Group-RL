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
    
        
        # Combine rewards with weights
        total_reward = coverage_reward
        return total_reward

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
        reward = self._calculate_reward()
        
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
    
