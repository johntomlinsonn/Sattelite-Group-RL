import pygame
import sys
import numpy as np

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
        # Draw grid (optional, for visual reference)
        for i in range(0, self.width, self.width // self.grid_width):
            pygame.draw.line(self.screen, (220,220,220), (i,0), (i,self.height))
        for j in range(0, self.height, self.height // self.grid_height):
            pygame.draw.line(self.screen, (220,220,220), (0,j), (self.width,j))
        # Draw satellites and coverage
        for sat in self.env.satellites:
            x, y = sat.getPosition()
            # Convert grid to screen coordinates
            sx = int(x * self.width / self.grid_width)
            sy = int(y * self.height / self.grid_height)
            # Draw coverage circle
            coverage_px = int(self.env.coverage_radius * self.width / self.grid_width)
            pygame.draw.circle(self.screen, (0, 0, 255, 50), (sx, sy), coverage_px, width=0)
            # Draw satellite
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

# Example usage:
# from environment import SatelliteSwarmEnv
# env = SatelliteSwarmEnv(num_satellites=10, grid_size=100, max_timesteps=200, coverage_radius=10)
# renderer = SatelliteRenderer(env)
# renderer.run()
