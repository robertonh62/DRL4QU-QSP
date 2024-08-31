# Quantum State Preparation and Unitary Matrix Synthesis Using Deep Reinforcement Learning

## Summary 
In this thesis, we explore the application of Deep Reinforcement Learning for Quantum State Preparation and Unitary Matrix Synthesis, focusing on using two reinforcement learning algorithms Deep Q-Networks (DQN) and Proximal Policy Optimization (PPO).

The approach is based on the work done in [Quantum compiling by deep reinforcement learning](https://www.nature.com/articles/s42005-021-00684-3) and [Self-correcting quantum many-body control using reinforcement learning with tensor networks](https://www.nature.com/articles/s42256-023-00687-5)

## Content
__Environment__
- [environment.Qsim.py](environment.Qsim.py): Contains a quantum computer simulator. It is built with the library Pytorch and is set up to be able to run on GPU.
- [environment.Qenv_unitary.py](environment.Qenv_unitary.py): Contains a Gym-style wrapper, to interact with the simulator as an environment for the Unitary Matrix Synthesis problem.
- [environment.Qenv_state.py](environment.Qenv_state.py): Gym-style wrapper for the Quantum State Preparation problem.
- [environment.Qenv_state_ising.py](environment.Qenv_state_ising.py): Gym-style wrapper for the Quantum State Preparation problem, applied to the ground state of the Ising Hamiltonian.

__Agents__
- [agents.agents.py](agents.agents.py): Contains the implementation of the DQN algorithm, with its variants Double DQN and Dueling DQN.

__Run__
- [Qrun_DQN.ipynb](Qrun_DQN.ipynb): Example of how to train a model using the available environments with the DQN algorithm.
- [Qrun_PPO.ipynb](Qrun_PPO.ipynb): Example of how to train a model using the available environments with the PPO algorithm.

## Requirements
The code is written in Python and besides the usual libraries (numpy, pandas, matplotblib), it also requires the following packages:
- [Pytorch](https://github.com/pytorch/pytorch): for the quantum computer simulation and the DQN implementation.
- [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3): For the out of the box PPO implementation.
- [Pennylane](https://github.com/PennyLaneAI/pennylane): visualization of circuits.
- [Seaborn](https://github.com/mwaskom/seaborn): format of plots.
