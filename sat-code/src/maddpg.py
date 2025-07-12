import torch
import numpy as np
import pygame
import sys
import time
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import os
from datetime import datetime

class MADDPGVisualizer:
    """
    A class to visualize the MADDPG training process in real-time.
    Shows satellite positions, coverage, and training metrics.
    """
    
    def __init__(self, env, width=1200, height=800, fps=30, save_dir=None):
        """
        Initialize the visualizer.
        
        Args:
            env: The satellite environment
            width: Screen width in pixels
            height: Screen height in pixels
            fps: Frames per second for visualization
            save_dir: Directory to save screenshots (None = don't save)
        """
        self.env = env
        self.width = width
        self.height = height
        self.fps = fps
        self.save_dir = save_dir
        
        # Initialize Pygame
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Satellite Swarm RL Visualization')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 14)
        self.title_font = pygame.font.SysFont('Arial', 24, bold=True)
        
        # Screen layout
        self.sim_width = int(self.width * 0.7)
        self.sim_height = self.height
        self.chart_width = self.width - self.sim_width
        self.chart_height = self.height
        
        # Training metrics
        self.episode_scores = []
        self.avg_scores = []
        self.episode_counter = 0
        self.noise_scale = 0.0
        self.current_score = 0.0
        self.best_score = -np.inf
        self.episode_durations = []
        self.start_time = time.time()
        
        # Create save directory if needed
        if self.save_dir:
            os.makedirs(self.save_dir, exist_ok=True)
            
        # Initialize matplotlib figure for metrics
        self.fig = Figure(figsize=(self.chart_width/100, self.chart_height/200), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        
    def update_metrics(self, episode, score, avg_score, noise_scale):
        """
        Update training metrics.
        
        Args:
            episode: Current episode number
            score: Score for the current episode
            avg_score: Average score over recent episodes
            noise_scale: Current exploration noise scale
        """
        self.episode_counter = episode
        self.current_score = score
        self.best_score = max(self.best_score, score)
        self.noise_scale = noise_scale
        
        # Update score history
        self.episode_scores.append(score)
        self.avg_scores.append(avg_score)
        
        # Update episode duration
        current_time = time.time()
        self.episode_durations.append(current_time - self.start_time)
        self.start_time = current_time
    
    def draw_metrics(self):
        """Draw training metrics on the right side of the screen."""
        # Clear the right panel
        pygame.draw.rect(
            self.screen, 
            (240, 240, 240), 
            (self.sim_width, 0, self.chart_width, self.height)
        )
        
        # Draw divider line
        pygame.draw.line(
            self.screen,
            (200, 200, 200),
            (self.sim_width, 0),
            (self.sim_width, self.height),
            2
        )
        
        # Draw title
        title = self.title_font.render("Training Progress", True, (0, 0, 0))
        self.screen.blit(title, (self.sim_width + 20, 20))
        
        # Draw episode info
        episode_text = self.font.render(f"Episode: {self.episode_counter}", True, (0, 0, 0))
        self.screen.blit(episode_text, (self.sim_width + 20, 60))
        
        # Draw score info
        current_score_text = self.font.render(f"Current Score: {self.current_score:.2f}", True, (0, 0, 0))
        self.screen.blit(current_score_text, (self.sim_width + 20, 80))
        
        best_score_text = self.font.render(f"Best Score: {self.best_score:.2f}", True, (0, 0, 0))
        self.screen.blit(best_score_text, (self.sim_width + 20, 100))
        
        # Draw exploration info
        noise_text = self.font.render(f"Exploration Noise: {self.noise_scale:.2f}", True, (0, 0, 0))
        self.screen.blit(noise_text, (self.sim_width + 20, 120))
        
        # Calculate elapsed time
        elapsed = time.time() - self.start_time
        elapsed_text = self.font.render(f"Episode Time: {elapsed:.1f}s", True, (0, 0, 0))
        self.screen.blit(elapsed_text, (self.sim_width + 20, 140))
        
        # Draw score chart if we have data
        if len(self.episode_scores) > 1:
            self.ax.clear()
            x = range(1, len(self.episode_scores) + 1)
            self.ax.plot(x, self.episode_scores, 'b-', label='Episode Score')
            
            if len(self.avg_scores) > 1:
                self.ax.plot(x, self.avg_scores, 'r-', label='Average Score')
                
            self.ax.set_xlabel('Episode')
            self.ax.set_ylabel('Score')
            self.ax.set_title('Training Progress')
            self.ax.legend(loc='lower right')
            self.ax.grid(True, linestyle='--', alpha=0.7)
            
            # Convert matplotlib figure to pygame surface
            self.canvas.draw()
            buf = self.canvas.buffer_rgba()
            chart_surface = pygame.image.frombuffer(buf, (int(self.chart_width * 0.9), int(self.chart_height * 0.4)), "RGBA")
            self.screen.blit(chart_surface, (self.sim_width + 20, 180))
    
    def draw_environment(self):
        """Draw the satellite environment."""
        # Clear the simulation area
        pygame.draw.rect(
            self.screen, 
            (255, 255, 255), 
            (0, 0, self.sim_width, self.sim_height)
        )
        
        # Draw title
        title = self.title_font.render("Satellite Swarm Simulation", True, (0, 0, 0))
        self.screen.blit(title, (20, 20))
        
        # Get grid dimensions
        grid_width = self.env.grid_width
        grid_height = self.env.grid_height
        
        # Calculate scaling factors
        scale_x = (self.sim_width - 100) / grid_width
        scale_y = (self.sim_height - 100) / grid_height
        
        # Draw grid (optional, for visual reference)
        for i in range(0, grid_width + 1, max(1, grid_width // 10)):
            x = 50 + i * scale_x
            pygame.draw.line(
                self.screen, 
                (220, 220, 220), 
                (x, 50), 
                (x, 50 + grid_height * scale_y)
            )
            
        for j in range(0, grid_height + 1, max(1, grid_height // 10)):
            y = 50 + j * scale_y
            pygame.draw.line(
                self.screen, 
                (220, 220, 220), 
                (50, y), 
                (50 + grid_width * scale_x, y)
            )
            
        # Draw satellites and coverage
        for sat in self.env.satellites:
            x, y = sat.getPosition()
            vx, vy = sat.velocity
            
            # Convert grid to screen coordinates
            sx = 50 + x * scale_x
            sy = 50 + y * scale_y
            
            # Draw coverage circle (semi-transparent)
            coverage_px = int(self.env.coverage_radius * scale_x)
            coverage_circle = pygame.Surface((coverage_px*2, coverage_px*2), pygame.SRCALPHA)
            pygame.draw.circle(coverage_circle, (0, 0, 255, 50), (coverage_px, coverage_px), coverage_px)
            self.screen.blit(coverage_circle, (sx - coverage_px, sy - coverage_px))
            
            # Draw satellite
            pygame.draw.circle(self.screen, (255, 0, 0), (int(sx), int(sy)), 5)
            
            # Draw velocity vector
            pygame.draw.line(
                self.screen,
                (0, 200, 0),
                (sx, sy),
                (sx + vx * 5, sy + vy * 5),
                2
            )
            
        # Draw coverage percentage
        coverage = self.env.calculate_coverage() * 100
        coverage_text = self.font.render(f"Earth Coverage: {coverage:.1f}%", True, (0, 0, 0))
        self.screen.blit(coverage_text, (50, self.sim_height - 30))
        
    def update(self):
        """Update the visualization."""
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
                
        # Draw the environment and metrics
        self.draw_environment()
        self.draw_metrics()
        
        # Update the display
        pygame.display.flip()
        self.clock.tick(self.fps)
        
    def save_screenshot(self, episode):
        """Save a screenshot of the current visualization."""
        if self.save_dir:
            filename = f"{self.save_dir}/episode_{episode:04d}.png"
            pygame.image.save(self.screen, filename)
            
    def close(self):
        """Close the visualization."""
        pygame.quit()


def create_visualizer(env, save_dir=None):
    """
    Create a visualization module for MADDPG training.
    
    Args:
        env: The satellite environment
        save_dir: Directory to save screenshots (None = don't save)
        
    Returns:
        visualizer: The MADDPGVisualizer instance
    """
    # Create a timestamped directory for saving visualizations
    if save_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        save_dir = f"visualizations/maddpg_{timestamp}"
        
    return MADDPGVisualizer(env, save_dir=save_dir)
