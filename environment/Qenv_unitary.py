from gymnasium import Env, spaces
from environment.Qsim import Circuit
import numpy as np
import torch as pt
from typing import Union, Dict, Tuple, Optional, Any
from collections import OrderedDict
from stable_baselines3.common.type_aliases import GymStepReturn
from scipy.stats import unitary_group

class Qenv_unitary_gym(Env, Circuit):
    def __init__(self, 
                 N: int,
                 goal: Optional[pt.Tensor]=None,            # If None it will genetate a sample of Haar unitary matrices
                 num_goals: Optional[int]=int(1e3),         # If target is None it wil say how many Matrices to generate
                 max_steps: int=10, 
                 fidelity_threshold: float=0.99,            # Fidelity = 1 - distance(State,Target)
                 fidelity_function: str='Fnorm',            # Options: 'Fnorm', 'Haar_integral', 'Hilbert_Schmidt'
                 qgates: Union[str, list]='HRC',            # Options: 'HRC', 'Pauli', 'clifford_T'
                 adjoint: bool=False,                       # If true it also takes adjoint gates
                 obs_type: str='Box',                       # Options: 'Box', 'Dict'
                 qubit_connectivity: Optional[list]=None):
        
        super().__init__(N=N)

        # Reward related stuff
        self.max_steps = max_steps
        self.fidelity_threshold = fidelity_threshold
        self.fidelity_function = getattr(self, fidelity_function, None)
        self.multiple_goal = goal == None
        self.goal = goal
        if self.multiple_goal == True:
            self.num_goals = num_goals
            self.goals = pt.tensor(unitary_group.rvs(dim=2**self.N, size=self.num_goals), dtype=pt.complex64, device=self.device)

        # Define action space
        if isinstance(qgates, str):
            qgates = getattr(self, qgates + '_gates', None)
            qgates = qgates + self.control_gates if self.N > 1 else qgates
        self.single_qgates = list(set(qgates) & set(self.single_qgates))
        self.double_qgates = list(set(qgates) & set(self.double_qgates))

        if qubit_connectivity == None:
            self.qubit_connectivity = self.qubit_combinations
        else:
            self.qubit_connectivity = qubit_connectivity

        self.explicit_actions = []
        for gate in self.single_qgates:
            for qubit in self.qubit_range:
                self.explicit_actions.append((gate, qubit, False))
                if adjoint == True:
                    self.explicit_actions.append((gate, qubit, True))

        for gate in self.double_qgates:
            for qubits in self.qubit_connectivity:
                self.explicit_actions.append((gate, qubits, False))

        self.action_space = spaces.Discrete(len(self.explicit_actions))

        # Define Observation Space
        self.obs_type = obs_type
        obs_box = spaces.Box(low=-1.0, high=1.0, shape=(pow(2, 2*self.N+1),), dtype=np.float32)
        if obs_type == 'Box':
            self.observation_space = obs_box
        elif obs_type == 'Dict':
            self.observation_space = spaces.Dict({'observation': obs_box, 'achieved_goal': obs_box, 'desired_goal': obs_box}, seed=42)
    
    def _get_obs(self) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        
        obs = self.desired_goal@pt.adjoint(self.Unitary)
        self.observation = obs
        obs = obs.cpu().flatten()
        obs = pt.concatenate([obs.real, obs.imag], dim=0).numpy()

        unitary = self.Unitary
        unitary = unitary.cpu().flatten()
        unitary = pt.concatenate([unitary.real, unitary.imag], dim=0).numpy()

        if self.obs_type == 'Dict':
            obs = OrderedDict(
                [
                    ("observation", obs.copy()),
                    ("achieved_goal", unitary.copy()),
                    ("desired_goal", self.desired_goal_np.copy()),
                ]
            )

        return obs
    
    def reset(self, seed=None)-> Tuple[Union[np.ndarray, Dict[str, np.ndarray]], Dict]:
        self.reset_circuit()
        self.fidelity = 0

        if self.multiple_goal == True:
            # Sample new target from unitaries
            sample = np.random.randint(self.num_goals)
            self.goal = self.goals[sample]
 
        self.desired_goal = self.goal
        
        desired_goal_np = self.desired_goal.cpu().flatten()
        self.desired_goal_np = pt.concatenate([desired_goal_np.real, desired_goal_np.imag], dim=0).numpy()

        observation = self._get_obs()
        return observation, {}

    def step(self, action: int) -> GymStepReturn:
        gate_symbol, qubit_idx, adjoint = self.explicit_actions[action]
        # Change so it don't uses get_gate anymore
        qgate = self.get_gate(gate_symbol, qubit_idx, adjoint=adjoint)
        self.Unitary = qgate @ self.Unitary

        self.state = qgate @ self.state
        self.history.loc[len(self.history)] = {'Gate':gate_symbol, 'Qubits':qubit_idx, 'Theta':0, 'Adjoint':adjoint}
        self.counter += 1

        observation = self._get_obs()
        if self.obs_type == 'Box':
            reward = float(self.compute_reward(self.Unitary, self.desired_goal, None).item())
        elif self.obs_type == 'Dict':
            reward = float(self.compute_reward(observation["achieved_goal"], observation["desired_goal"], None).item())
        done = reward == 0
        truncated = self.counter >= self.max_steps
        info = {"is_success": done, "Fidelity": self.fidelity.item()}

        return observation, reward, done, truncated, info

    def compute_reward(self, achieved_goal: Union[pt.Tensor, np.ndarray], desired_goal: Union[pt.Tensor, np.ndarray], 
                       _info: Optional[Dict[str, Any]]) -> np.float32:
        if self.obs_type == 'Dict':
            achieved_goal = self.np2tensor(achieved_goal)
            desired_goal = self.np2tensor(desired_goal)

        self.fidelity = self.fidelity_function(achieved_goal, desired_goal)
        reward = -(self.fidelity < self.fidelity_threshold).astype(np.float32)
        return reward

    def np2tensor(self, nparray: np.ndarray) -> pt.Tensor:
        if nparray.ndim == 1:
            half = int(nparray.shape[0]/2)
            size = int(np.sqrt(half))
            tensor = pt.tensor((nparray[:half] + 1j * nparray[half:]).reshape(size, size), dtype=pt.complex64, device=self.device)
        else:
            # batch cases
            half = int(nparray.shape[1]/2)
            size = int(np.sqrt(half))
            tensor = pt.tensor((nparray[:,:half] + 1j * nparray[:,half:]).reshape(nparray.shape[0],size, size), dtype=pt.complex64, device=self.device)
        return tensor

    def render(self) -> None:
        self.draw_circuit()

    def close(self) -> None:
        pass

    # Fidelity FUNCTIONS
    # 0 -> 1 (1 is when both Unitaries are the same)
    def Fnorm(self, achieved_goal: pt.Tensor, desired_goal: pt.Tensor) -> np.float32:
        max_axis = achieved_goal.ndim
        frobenius_norm = np.linalg.norm(achieved_goal - desired_goal, ord='fro',axis=(max_axis - 2, max_axis - 1))/2**(self.N+1)
        # F norms gives us distance so Fidelity = 1 - distance
        return 1 - frobenius_norm
    
    def Haar_integral(self, achieved_goal:pt.Tensor, desired_goal:pt.Tensor):
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

    def Hilbert_Schmidt(self, achieved_goal: pt.Tensor, desired_goal: pt.Tensor):
            # Note: Still don't work with batches
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