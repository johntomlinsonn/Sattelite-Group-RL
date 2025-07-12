import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import os
import time

from src.environment import SatelliteSwarmEnv
from src.agents import MADDPG, get_device_info, print_gpu_memory_usage
from src.maddpg import create_visualizer

def train_maddpg(
    env, 
    state_size, 
    action_size, 
    num_agents, 
    n_episodes=2000, 
    max_steps=1000, 
    batch_size=128,
    buffer_size=100000,
    gamma=0.99,
    tau=0.01,
    lr_actor=1e-4,
    lr_critic=1e-3,
    weight_decay=0,
    noise_scale_start=1.0,
    noise_scale_end=0.1,
    noise_scale_decay=0.99,
    print_every=10,
    save_every=100,
    visualize=True,
    visualization_fps=30,
    periodic_vis_interval=15
):
    """
    Train MADDPG agents in the given environment.
    
    Args:
        env: The environment to train in
        state_size: Size of each agent's state space
        action_size: Size of each agent's action space
        num_agents: Number of agents in the environment
        n_episodes: Number of episodes to train
        max_steps: Maximum steps per episode
        batch_size: Batch size for training
        buffer_size: Size of the replay buffer
        gamma: Discount factor
        tau: Soft update parameter
        lr_actor: Learning rate for the actor
        lr_critic: Learning rate for the critic
        weight_decay: L2 weight decay
        noise_scale_start: Starting scale for exploration noise
        noise_scale_end: Ending scale for exploration noise
        noise_scale_decay: Decay rate for exploration noise
        print_every: How often to print progress
        save_every: How often to save the model
        visualize: Whether to visualize the training continuously
        visualization_fps: Frames per second for visualization
        periodic_vis_interval: Show visualization every N episodes (even when visualize=False)
        
    Returns:
        scores: List of scores from each episode
    """
    # Create MADDPG trainer
    maddpg = MADDPG(
        state_size=state_size,
        action_size=action_size,
        num_agents=num_agents,
        buffer_size=buffer_size,
        batch_size=batch_size,
        gamma=gamma,
        tau=tau
    )
    
    # Print device information
    device_info = get_device_info()
    print(f"Training on device: {device_info['device']}")
    if device_info['cuda_available']:
        print(f"GPU: {device_info['device_name']}")
        print(f"Total GPU Memory: {device_info['total_memory']:.2f} GB")
        print_gpu_memory_usage()
    
    # Initialize score tracking
    scores = []
    scores_window = []  # Last 100 episodes
    score_best = -np.inf
    noise_scale = noise_scale_start
    
    # Create directory for saving models
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    save_dir = f"models/maddpg_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Set up visualization if enabled (or for periodic visualization)
    visualizer = None
    vis_dir = None
    if visualize or periodic_vis_interval > 0:
        vis_dir = f"visualizations/maddpg_{timestamp}"
        os.makedirs(vis_dir, exist_ok=True)
        visualizer = create_visualizer(env, save_dir=vis_dir)
        
        if not visualize:
            print(f"Periodic visualization enabled: will show every {periodic_vis_interval} episodes")
    
    # Training loop
    print("Starting training...")
    start_time = time.time()
    
    for episode in range(1, n_episodes+1):
        # Reset environment
        states, _ = env.reset()  # Using gymnasium API
        # No need to flatten states - environment now returns flattened observations
        
        episode_rewards = np.zeros(num_agents)
        
        for step in range(max_steps):
            # Get actions from all agents
            actions = maddpg.act(states, add_noise=True, noise_scale=noise_scale)
            
            # Take actions in the environment
            next_states, rewards, dones, _, _ = env.step(actions)  # Using gymnasium API
            # No need to flatten next_states - environment now returns flattened observations
            
            # Store the experience
            maddpg.step(states, actions, rewards, next_states, dones)
            
            # Update state and rewards
            states = next_states
            episode_rewards += rewards
            
            # Break if any agent is done
            if np.any(dones):
                break
                
            # Update visualization if enabled (continuous or periodic)
            should_visualize_step = False
            if visualizer is not None:
                if visualize:
                    # Continuous visualization mode
                    should_visualize_step = (step % 5 == 0)
                elif periodic_vis_interval > 0 and episode % periodic_vis_interval == 0:
                    # Periodic visualization mode - show every step during special episodes
                    should_visualize_step = True
                
                if should_visualize_step:
                    visualizer.update()
        
        # Decay noise for exploration
        noise_scale = max(noise_scale_end, noise_scale * noise_scale_decay)
        
        # Track scores
        score = np.mean(episode_rewards)
        scores.append(score)
        scores_window.append(score)
        
        # Update visualization metrics and handle periodic visualization
        if visualizer is not None:
            avg_score = np.mean(scores_window[-min(len(scores_window), 100):])
            visualizer.update_metrics(episode, score, avg_score, noise_scale)
            
            # Always update metrics, but only update display for continuous or periodic episodes
            if visualize:
                # Continuous visualization
                visualizer.update()
            elif periodic_vis_interval > 0 and episode % periodic_vis_interval == 0:
                # Periodic visualization - show this episode
                print(f"\n--- Periodic Visualization (Episode {episode}) ---")
                visualizer.update()
                print("--- End Periodic Visualization ---\n")
            
            # Save screenshot at intervals or periodic episodes
            should_save_screenshot = (
                (episode % save_every == 0 or episode == 1) or
                (not visualize and periodic_vis_interval > 0 and episode % periodic_vis_interval == 0)
            )
            if should_save_screenshot:
                visualizer.save_screenshot(episode)
        
        # Print progress
        if episode % print_every == 0:
            elapsed = time.time() - start_time
            avg_score = np.mean(scores_window[-print_every:])
            vis_mode = "continuous" if visualize else f"periodic (every {periodic_vis_interval})"
            print(f"Episode {episode}/{n_episodes} | Avg Score: {avg_score:.2f} | Noise: {noise_scale:.2f} | Time: {elapsed:.1f}s | Vis: {vis_mode}")
            
            # Print GPU memory usage if available
            if device_info['cuda_available']:
                print_gpu_memory_usage()
            
            # Plot progress
            plt.figure(figsize=(10, 5))
            plt.plot(np.arange(len(scores)), scores)
            plt.ylabel('Score')
            plt.xlabel('Episode')
            plt.title(f'MADDPG Training Scores (Avg: {avg_score:.2f})')
            plt.savefig(f"{save_dir}/progress.png")
            plt.close()
        
        # Save best model
        if score > score_best and episode > 100:
            score_best = score
            maddpg.save(f"{save_dir}/best")
            print(f"New best model saved with score {score_best:.2f}")
        
        # Save checkpoint
        if episode % save_every == 0:
            maddpg.save(f"{save_dir}/checkpoint_{episode}")
            print(f"Checkpoint saved at episode {episode}")
    
    # Save final model
    maddpg.save(f"{save_dir}/final")
    print(f"Final model saved after {n_episodes} episodes")
    
    # Close visualizer if enabled
    if visualizer is not None:
        visualizer.close()
    
    # Calculate total training time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Training completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Plot final learning curve
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(scores)), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode')
    plt.title('MADDPG Training Scores')
    plt.savefig(f"{save_dir}/final_scores.png")
    plt.close()
    
    return scores

def train_and_visualize(visualize=True, periodic_vis_interval=15):
    """Main function to set up and run the training process."""
    # Create environment
    num_satellites = 3
    grid_size = 100
    max_timesteps = 200
    coverage_radius = 10
    
    env = SatelliteSwarmEnv(
        num_satellites=num_satellites, 
        grid_size=grid_size, 
        max_timesteps=max_timesteps, 
        coverage_radius=coverage_radius
    )
    
    # Determine state and action dimensions
    state_size = 4  # Each satellite has position (x,y) and velocity (vx,vy)
    action_size = 2  # Each satellite controls acceleration in x and y directions
    
    # Train agents
    scores = train_maddpg(
        env=env,
        state_size=state_size,
        action_size=action_size,
        num_agents=num_satellites,
        n_episodes=2000,
        max_steps=max_timesteps,
        print_every=10,
        save_every=100,
        visualize=visualize,
        periodic_vis_interval=periodic_vis_interval
    )
    
    return scores

if __name__ == "__main__":
    train_and_visualize()
