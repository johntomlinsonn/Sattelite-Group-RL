from src.train import train_and_visualize
from src.agents import get_device_info
import argparse

def main():
    """
    Main entry point for the Satellite Swarm Reinforcement Learning project.
    Trains a MADDPG agent to control multiple satellites for optimal coverage.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Satellite Swarm RL Training')
    parser.add_argument('--no-vis', action='store_true', help='Disable continuous visualization (still shows every 15 episodes)')
    parser.add_argument('--test-cuda', action='store_true', help='Test CUDA setup and exit')
    parser.add_argument('--periodic-interval', type=int, default=15, 
                       help='Show visualization every N episodes when --no-vis is used (default: 15)')
    args = parser.parse_args()
    
    # Test CUDA if requested
    if args.test_cuda:
        from test_cuda import test_cuda_setup
        test_cuda_setup()
        return
    
    # Show device information
    device_info = get_device_info()
    print("=" * 60)
    print("Satellite Swarm RL Training with MADDPG")
    print("=" * 60)
    print(f"Device: {device_info['device']}")
    if device_info['cuda_available']:
        print(f"GPU: {device_info['device_name']}")
        print(f"GPU Memory: {device_info['total_memory']:.2f} GB")
    
    # Show visualization mode info
    if not args.no_vis:
        print("Visualization: Continuous (real-time)")
    else:
        print(f"Visualization: Periodic (every {args.periodic_interval} episodes)")
    print("=" * 60)
    
    print("Starting training...")
    train_and_visualize(
        visualize=not args.no_vis, 
        periodic_vis_interval=args.periodic_interval if args.no_vis else 0
    )
    print("Training complete!")

if __name__ == "__main__":
    main()
