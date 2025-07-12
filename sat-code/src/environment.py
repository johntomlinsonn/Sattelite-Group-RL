import gymnasium, numpy as np, random as rand
import pygame
import sys

class SatelliteSwarmEnv(gymnasium.Env):
    def __init__(self,num_satellites,grid_size,max_timesteps,coverage_radius):
        super(SatelliteSwarmEnv, self).__init__()
        self.num_satellites = num_satellites
        self.max_timesteps = max_timesteps
        self.coverage_radius = coverage_radius
    
        self.grid_width = grid_size
        self.grid_height = grid_size
        self.earth_coverage_map = np.zeros((self.grid_width, self.grid_height), dtype=bool)

        self.satellites = [Sat(self.grid_width, self.grid_height) for _ in range(num_satellites)]
        
        # Define action and observation spaces
        self.action_space = gymnasium.spaces.Discrete(num_satellites * 5)
        self.observation_space = gymnasium.spaces.Box(
            low=0, 
            high=grid_size, 
            shape=(num_satellites, 2), 
            dtype=np.float32
        )
    def reset(self, *, seed=None, options=None):
        return super().reset(seed=seed, options=options)
    
    def calculate_coverage(self):
        self.earth_coverage_map.fill(False)
        for sat in self.satellites:
            x, y = sat.getPosition()
            for i in range(max(0, x - self.coverage_radius), min(self.grid_width, x + self.coverage_radius + 1)):
                for j in range(max(0, y - self.coverage_radius), min(self.grid_height, y + self.coverage_radius + 1)):
                    if (i - x) ** 2 + (j - y) ** 2 <= self.coverage_radius ** 2:
                        self.earth_coverage_map[i, j] = True
        return np.sum(self.earth_coverage_map) / (self.grid_width * self.grid_height)
    
    def step(self):
        # Update satellite positions based on their velocities
        for sat in self.satellites:
            x, y = sat.getPosition()
            vx, vy = sat.velocity
        
            # Update position
            new_x = max(0, min(self.grid_width - 1, x + vx))
            new_y = max(0, min(self.grid_height - 1, y + vy))
            sat.position = (new_x, new_y)
            for sat in self.satellites:
                print(earth.calculate_coverage())
    
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
            pygame.draw.line(self.screen, (220,220,220), (i,0), (i,self.height))
        for j in range(0, self.height, self.height // self.grid_height):
            pygame.draw.line(self.screen, (220,220,220), (0,j), (self.width,j))
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
        for _ in range(steps):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.env.step()
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
        self.velocity = (rand.randint(-10,10), rand.randint(-10,10))

    def updateVelocity(self, velocity_x, velocity_y):
        self.velocity = (velocity_x, velocity_y)
    
    def randomizeVelocity(self):
        self.velocity = (rand.randint(-10, 10), rand.randint(-10, 10))

    def getPosition(self):
        return self.position
    
    
earth = SatelliteSwarmEnv(num_satellites=10, grid_size=100, max_timesteps=200, coverage_radius=10)
print(earth.calculate_coverage())
renderer = SatelliteRenderer(earth)
renderer.run(steps=500, fps=2)