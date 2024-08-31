from gymnasium import Env, spaces
from environment.Qsim import Circuit
import numpy as np
import torch as pt
from typing import Union, Dict, Tuple, Optional, Any
from collections import OrderedDict
from stable_baselines3.common.type_aliases import GymStepReturn
from scipy.stats import unitary_group
import itertools

class Qenv_unitary_gym(Env, Circuit):
    """
    Gym style environment for aproximating unitaries in Quantum Circuits
    
    Parameters:
    N:                      int
        Number of Qubits
    goals:                  Tensor | None
        If is a tensor is the unitary matrix to aproximate, if is None generate goals using Haar matrices
    num_goals:              int | None
        If goals is a None it says how many unitaries to generate (if is -1 it generatate a new goal each restart)
    max_steps:              int
        Max number of gates the circuit can apply 
    fidelity_threshold:     float  
        Threshold fidelity limit to concider a episode as succesfull
    fidelity_function:      str
        Function that is used to calculate the fidelity between the two unitaries
        Options: 'Fnorm', 'Haar_integral', 'Hilbert_Schmidt'
    sparse_reward:          bool
        For True it uses Sparse reward (r = -(F(t)<F_threshold)) for False dense reward (r = log(F(t)))
    qgates:                 str | list
        Quantum gates available to the circuit. If is list the elements are the names of the gates.
        If is a string is the set of gates from known ones. 
        Options: 'HRC', 'pauli', 'clifford_T'
    adjoint:                bool                           
        If true it also takes adjoint gates
    qubit_connectivity:     list | None
        Qubit conections for two qubit gates, if None all qubits conections are available
    device:                 str | None
        Device (cpu, cuda, …) on which the code should be run. If None, the code will be run on the GPU if possible 
    """

    def __init__(self,
                 N: int,
                 goals: Optional[pt.Tensor]=None,
                 num_goals: Optional[int]=-1,
                 max_steps: int=10, 
                 fidelity_threshold: float=0.99,
                 fidelity_function: str='Haar_integral',
                 sparse_reward: bool=False,
                 qgates: Union[str, list]='HRC',
                 adjoint: bool=False,
                 qubit_connectivity: Optional[list]=None,
                 device: Optional[str]=None):
        
        super().__init__(N=N, device=device)

        self.max_steps = max_steps

        # DEFINE ACTION SPACE
        # Get set of single and two qubit gates
        if isinstance(qgates, str):
            qgates = getattr(self, qgates + '_gates', None)
            qgates = qgates + self.control_gates if self.N > 1 else qgates
        self.qgates = qgates
        self.single_qgates = list(set(qgates) & set(self.single_qgates))
        self.double_qgates = list(set(qgates) & set(self.double_qgates))

        # Get qubits connectivities
        if qubit_connectivity == None:
            self.qubit_connectivity = self.qubit_combinations
        else:
            self.qubit_connectivity = qubit_connectivity

        # Get set of all posible actions
        self.explicit_actions = []
        for gate in self.single_qgates:
            for qubit in self.qubit_range:
                self.explicit_actions.append((gate, qubit, adjoint==True))

        for gate in self.double_qgates:
            for qubits in self.qubit_connectivity:
                self.explicit_actions.append((gate, qubits, False))

        self.action_space = spaces.Discrete(len(self.explicit_actions))

        # DEFINE OBSERVATION SPACE
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(pow(2, 2*self.N+1),), dtype=np.float32)

        # DEFINE REWARD
        self.fidelity_threshold = fidelity_threshold
        self.fidelity_function = getattr(self, fidelity_function, None)
        self.sparse_reward = sparse_reward
        if isinstance(goals, pt.Tensor):
            self.goals = goals
            self.num_goals = goals.shape[0] if goals.ndim == 3 else 1
        elif num_goals != -1:
            self.goals = pt.tensor(unitary_group.rvs(dim=2**self.N, size=num_goals), dtype=pt.complex64, device=self.device)
            self.num_goals = num_goals
        elif num_goals == -1:
            self.num_goals = num_goals

    def _get_obs(self):
        # Observation the agent will return after each step (obs = U V^t)
        obs = self.goal @ pt.adjoint(self.Unitary)
        self.observation = obs
        obs = obs.cpu().flatten()
        obs = pt.concatenate([obs.real, obs.imag], dim=0).numpy()
        return obs
    
    def reset(self, seed=None):
        self.reset_circuit()
        self.score = 0

        # Sample goal
        if self.num_goals == -1:
            self.goal = pt.tensor(unitary_group.rvs(dim=2**self.N, size=1), dtype=pt.complex64, device=self.device)
        elif self.num_goals == 1:
            self.goal = self.goals
        else:
            sample = np.random.randint(self.num_goals)
            self.goal = self.goals[sample]

        # Get observation
        observation = self._get_obs()
        return observation, {}

    def step(self, action):
        gate_symbol, qubit_idx, adjoint = self.explicit_actions[action]
        qgate = self.get_gate(gate_symbol=gate_symbol, qubit=qubit_idx, adjoint=adjoint)
        self.apply_gate(gate=qgate, gate_symbol=gate_symbol, qubits=qubit_idx, adjoint=adjoint)

        observation = self._get_obs()
        # Reward calculation
        reward = float(self.compute_reward(self.Unitary, self.goal, None).item())
        self.score += reward
        # Done calculation
        done = self.fidelity > self.fidelity_threshold
        # Truncated and info calculation
        truncated = self.counter >= self.max_steps
        info = {'Fidelity': self.fidelity.item()}
        if (done == True) or (truncated == True):
            info['is_success'] = done
            info['l'] = self.counter
            info['r'] = self.score
            info['Goal'] = self.goal.tolist()
            info['Circuit'] = list(self.history.itertuples(index=False, name=None))

        return observation, reward, done, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, _info):
        self.fidelity = self.fidelity_function(achieved_goal, desired_goal)
        if self.sparse_reward == True:
            reward = -(self.fidelity < self.fidelity_threshold).astype(np.float32)
        else:
            reward = np.log(self.fidelity + 1e-20)
        return reward

    def render(self):
        self.draw_circuit()

    def close(self):
        pass

    # FIDELITY FUNCTIONS
    # 0 -> 1 (1 is when both Unitaries are the same)
    def Fnorm(self, achieved_goal, desired_goal):
        max_axis = achieved_goal.ndim
        frobenius_norm = np.linalg.norm(achieved_goal - desired_goal, ord='fro',axis=(max_axis - 2, max_axis - 1))/2**(self.N+1)
        # F norms gives us distance so Fidelity = 1 - distance
        return 1 - frobenius_norm
    
    def Haar_integral(self, achieved_goal, desired_goal):
        size = 10000
        
        haar_matrices = pt.tensor(unitary_group.rvs(dim=2**self.N, size=size), dtype=pt.complex64, device=self.device)
        zero_state = pt.zeros(2**self.N, dtype=pt.complex64, device=self.device).reshape(-1,1)
        zero_state[0] = 1
        psi = haar_matrices @ zero_state
        psi_T = pt.adjoint(psi)
        U = pt.adjoint(achieved_goal) @ desired_goal

        if achieved_goal.ndim == 2:
            F = pt.abs((psi_T @ U @ psi).flatten())**2
            integral = pt.mean(F).cpu().numpy().astype(np.float32)

        else:
            bra = psi_T.unsqueeze(0) 
            bra = bra.expand(achieved_goal.shape[0], -1, -1, -1) 

            U = U.unsqueeze(1)

            ket = psi.unsqueeze(0)  
            ket = ket.expand(achieved_goal.shape[0], -1, -1, -1)  

            braUket = bra @ U @ ket

            F = pt.abs(braUket.squeeze(-1).squeeze(-1))**2
            integral = pt.mean(F, dim=1).cpu().numpy().astype(np.float32)

        return integral

    def Hilbert_Schmidt(self, achieved_goal, desired_goal):
            eval_sim = Circuit(N=self.N*2)

            for i in range(1,self.N+1):
                eval_sim.Hadamard(i)

            for i in range(1,self.N+1):
                eval_sim.CNOT([i, i+self.N])

            eval_sim.Unitary_gate(desired_goal, qubit=[1,self.N], gate_name='Target')
            eval_sim.Unitary_gate(pt.conj(achieved_goal), qubit=[self.N+1,self.N*2], gate_name='Compiled^*')

            for i in range(self.N, 0, -1):
                eval_sim.CNOT([i, i+self.N])

            for i in range(1,self.N+1):
                eval_sim.Hadamard(i)

            zero_state_prob = pt.abs(eval_sim.state[0])**2
            return zero_state_prob