"""
Test script to verify CUDA setup and device detection for the MADDPG training.
This script will check if CUDA is available and test basic tensor operations.
"""

import torch
import numpy as np
from src.agents import get_device_info, print_gpu_memory_usage, device

def test_cuda_setup():
    """Test CUDA setup and basic operations."""
    print("=" * 50)
    print("CUDA Setup Test")
    print("=" * 50)
    
    # Get device information
    device_info = get_device_info()
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {device_info['cuda_available']}")
    print(f"Current device: {device_info['device']}")
    
    if device_info['cuda_available']:
        print(f"GPU: {device_info['device_name']}")
        print(f"Total GPU Memory: {device_info['total_memory']:.2f} GB")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Test basic tensor operations
        print("\nTesting tensor operations...")
        
        # Create test tensors
        x = torch.randn(1000, 1000, device=device)
        y = torch.randn(1000, 1000, device=device)
        
        print_gpu_memory_usage()
        
        # Matrix multiplication
        start_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        end_time = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
        
        if torch.cuda.is_available():
            start_time.record()
        
        z = torch.mm(x, y)
        
        if torch.cuda.is_available():
            end_time.record()
            torch.cuda.synchronize()
            elapsed_time = start_time.elapsed_time(end_time)
            print(f"Matrix multiplication (1000x1000) completed in {elapsed_time:.2f} ms")
        
        print_gpu_memory_usage()
        
        # Test neural network operations
        print("\nTesting neural network operations...")
        from src.agents import Actor, Critic
        
        # Create test networks
        actor = Actor(state_size=4, action_size=2).to(device)
        critic = Critic(state_size=4, action_size=2, num_agents=3).to(device)
        
        # Test forward pass
        test_state = torch.randn(32, 4, device=device)  # Batch of states
        test_actions = torch.randn(32, 6, device=device)  # Batch of actions for all agents
        test_states_all = torch.randn(32, 12, device=device)  # All agent states
        
        actor_output = actor(test_state)
        critic_output = critic(test_states_all, test_actions)
        
        print(f"Actor output shape: {actor_output.shape}")
        print(f"Critic output shape: {critic_output.shape}")
        print(f"Networks created and tested successfully!")
        
        print_gpu_memory_usage()
        
        # Clean up
        del x, y, z, actor, critic, test_state, test_actions, test_states_all, actor_output, critic_output
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        print("\nMemory cleaned up.")
        print_gpu_memory_usage()
        
    else:
        print("CUDA not available. Training will use CPU.")
        print("For better performance, consider using a system with CUDA-compatible GPU.")
        
        # Test CPU operations
        print("\nTesting CPU tensor operations...")
        x = torch.randn(100, 100)
        y = torch.randn(100, 100)
        z = torch.mm(x, y)
        print("CPU tensor operations working correctly!")
    
    print("\n" + "=" * 50)
    print("CUDA Setup Test Complete")
    print("=" * 50)

if __name__ == "__main__":
    test_cuda_setup()
