"""
Training script for the 3D satellite swarm environment.
"""

import torch
import yaml
from envs import SatelliteEnv3D
from agents.maddpg.maddpg_agent import MADDPGAgent
