from src.train import train_and_visualize
import argparse

def main():
    """
    Main entry point for the Satellite Swarm Reinforcement Learning project.
    Trains a MADDPG agent to control multiple satellites for optimal coverage.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Satellite Swarm RL Training')
    parser.add_argument('--no-vis', action='store_true', help='Disable visualization')
    args = parser.parse_args()
    
    print("Starting Satellite Swarm RL Training...")
    train_and_visualize(visualize=not args.no_vis)
    print("Training complete!")

if __name__ == "__main__":
    main()
