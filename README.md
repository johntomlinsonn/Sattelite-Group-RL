# Traffic Signal Control Using Reinforcement Learning

This repository documents the planned development of a research project focused on applying reinforcement learning (RL) to traffic signal control using the Simulation of Urban MObility (SUMO) platform and the PyTorch deep learning framework. The goal of the project is to optimize traffic light timing to reduce congestion and waiting times in urban environments.

The project is currently in the planning stage. This README outlines the proposed methodology, roadmap, and tools for future development.

## Project Overview

The central objective of this project is to design and train a reinforcement learning agent capable of controlling traffic signals within a simulated environment. The simulation is powered by SUMO, with the Python TraCI API used for programmatic control. The agent will observe traffic conditions and learn to minimize metrics such as average vehicle waiting time and queue lengths.

The project is intended for research and educational purposes.

## Roadmap

The following is a high-level breakdown of the planned phases of the project:

### Step 1: Preliminary Research

- Study basic concepts in traffic signal control and reinforcement learning.
- Understand how SUMO operates and how it can be interfaced with external programs using TraCI.

**Suggested Reading:**
- [Traffic Signal Control - Wikipedia](https://en.wikipedia.org/wiki/Traffic-light_control_and_coordination)
- [OpenAI Spinning Up - Introduction to RL](https://spinningup.openai.com/en/latest/)
- [SUMO Quick Start](https://sumo.dlr.de/docs/Tutorials/Quick_Start.html)

### Step 2: Environment Setup

- Install SUMO on a local machine (Windows, macOS, or Linux).
- Run basic simulations and become familiar with the SUMO graphical user interface.
- Manually control traffic lights within SUMO to gain intuition.

**Resources:**
- [SUMO Installation Guide](https://sumo.dlr.de/docs/Installing.html)
- [SUMO GUI Overview](https://sumo.dlr.de/docs/SUMO-GUI.html)

### Step 3: Python Integration via TraCI

- Learn to use the TraCI Python API to interface with SUMO.
- Extract traffic metrics such as queue length and waiting time.
- Modify traffic signal phases using Python.

**Resources:**
- [TraCI Python API Documentation](https://sumo.dlr.de/docs/TraCI.html)
- [TraCI Tutorial](https://sumo.dlr.de/docs/Tutorials/TraCI.html)

### Step 4: Reinforcement Learning Environment Design

- Construct a Python-based RL environment that wraps SUMO using the Gym API style.
- Define:
  - State space (e.g., queue lengths, waiting times)
  - Action space (e.g., traffic phase switching)
  - Reward function (e.g., negative total waiting time)

**Reference Material:**
- [Creating Custom Gym Environments](https://www.gymlibrary.dev/content/environment_creation/)
- [Example Project - sumo-rl](https://github.com/LucasAlegre/sumo-rl)

### Step 5: Agent Implementation

- Develop the RL agent using PyTorch.
- Implement algorithms such as DQN, PPO, or A2C.
- Connect the agent’s outputs to SUMO actions via TraCI.

**Resources:**
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [RL in PyTorch - Intermediate Tutorial](https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html)
- [Stable-Baselines3 PPO Codebase](https://github.com/DLR-RM/stable-baselines3)

### Step 6: Training and Evaluation

- Run training over multiple episodes and evaluate convergence.
- Track metrics including average waiting time and total delay.
- Tune hyperparameters such as learning rate and discount factor.

### Step 7: Analysis and Visualization

- Visualize learned policies using the SUMO GUI.
- Plot learning curves using Matplotlib or TensorBoard.
- Compare against baseline control strategies (e.g., fixed-timing, random).

### Step 8: Experimentation

- Extend the project to multi-intersection traffic networks using multi-agent reinforcement learning.
- Explore different road layouts and reward strategies.

## Tools and Technologies

- **SUMO**: Traffic simulation platform
- **Python 3.x**
- **PyTorch**: Neural network and deep RL implementation
- **TraCI**: Python API to control SUMO
- **Matplotlib** and **TensorBoard** for visualization
- **VS Code**, **PyCharm**, or **Jupyter** for development

## Future Goals

- Extend to multi-agent learning across a grid of intersections
- Integrate real-world traffic data (if available)
- Explore transfer learning or curriculum learning strategies

## Related Work and Inspiration

- [sumo-rl: Reinforcement Learning on SUMO](https://github.com/LucasAlegre/sumo-rl)
- [Flow: Deep RL for Traffic Control](https://flow-project.github.io/)

## Contributing

This project is under initial development and not yet open for contributions. Once a baseline implementation is in place, contributions, bug reports, and feature suggestions will be welcome.

## License

This project is released under the MIT License. See the LICENSE file in the repository for full terms and conditions.

## Contact

If you are a student or researcher interested in collaborating or following this project’s development, please feel free to reach out via the Issues section or fork the repository for independent exploration.
