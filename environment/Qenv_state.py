from gymnasium import Env, spaces
from environment.Qsim import Circuit
import numpy as np
import torch as pt
from typing import Union, Dict, Tuple, Optional, Any
from collections import OrderedDict
from stable_baselines3.common.type_aliases import GymStepReturn
from scipy.stats import unitary_group
import itertools

class Qenv_state_gym(Env, Circuit):
    """
    Gym style environment for quantum state preparation in Quantum Circuits
    
    Parameters:
    N:                      int
        Number of Qubits
    goals:                  Tensor | None
        If is a tensor is the unitary matrix to aproximate, if is None generate goals using Haar matrices
    num_goals:              int | None
        If goals is a Nonee it says how many unitaries to generate (if is -1 it generatate a new goal each restart)
    max_steps:              int
        Max number of gates the circuit can apply 
    fidelity_threshold:     float  
        Threshold fidelity limit to concider a episode as succesfull
    sparse_reward:          bool
        For True it uses Sparse reward (r = -(F(t)<F_threshold) ) for False dense reward (r = F(t) - F(t-1))
    qgates:                 str | list
        Quantum gates available to the circuit. If is list the elements are the names of the gates.
        If is a string is the set of gates from known ones. Options: 'HRC', 'pauli', 'clifford_T'
    adjoint:                bool                           
        If true it also takes adjoint gates
    inverse_circuit:        bool
        If True we start from goal state and go to zero state
        If False we start from zero state and go to goal state
    obs_type:               str                           
        If the observation is just the current state, or the state concatenated with the goal
        Options: 'Single', 'Concat'
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
                 sparse_reward: bool=False,
                 qgates: Union[str, list]='HRC',
                 adjoint: bool=False,
                 inverse_circuit: bool=False,
                 obs_type: str='Single',
                 qubit_connectivity: Optional[list]=None,
                 device: Optional[str]=None):
        
        super().__init__(N=N,device=device)

        self.inverse_circuit = inverse_circuit
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
                #if adjoint == True:
                #    self.explicit_actions.append((gate, qubit, True))

        for gate in self.double_qgates:
            for qubits in self.qubit_connectivity:
                self.explicit_actions.append((gate, qubits, False))

        self.action_space = spaces.Discrete(len(self.explicit_actions))

        # DEFINE OBSERVATION SPACE
        self.obs_type = obs_type
        if obs_type == 'Single':
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2**(self.N+1),), dtype=np.float32)
        elif obs_type == 'Concat':
            self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2**(self.N+2),), dtype=np.float32)
        #elif obs_type == 'Dict':
        #    self.observation_space = spaces.Dict({'achieved_goal': obs_box, 'desired_goal': obs_box}, seed=42)

        # DEFINE REWARD
        self.fidelity_threshold = fidelity_threshold
        self.sparse_reward = sparse_reward
        if isinstance(goals, pt.Tensor):
            self.goals = goals
            self.num_goals = goals.shape[0] if goals.ndim == 3 else 1
        else:
            self.num_goals = num_goals
            if self.num_goals != -1:
                self.goals = self.haar_states(num_states=num_goals)

    def haar_states(self, num_states):
        haar_matrices = pt.tensor(unitary_group.rvs(dim=2**self.N, size=num_states), dtype=pt.complex64, device=self.device)
        zero_state = pt.zeros(2**self.N, dtype=pt.complex64, device=self.device).reshape(-1,1)
        zero_state[0] = 1
        random_states = haar_matrices @ zero_state
        return random_states

    def _get_obs(self):
        # Observation the agent will return after each step
        obs = self.state
        self.observation = obs
        obs = obs.cpu().reshape(-1,)
        obs = pt.concatenate([obs.real, obs.imag], dim=0).numpy()

        if self.obs_type == 'Concat':
            obs = np.concatenate((obs, self.desired_goal_np.copy()), axis=0)
        #elif self.obs_type == 'Dict':
        #    obs = OrderedDict(
        #        [
        #            ("achieved_goal", obs.copy()),
        #            ("desired_goal", self.desired_goal_np.copy()),
        #        ]
        #    )
        return obs
    
    def reset(self, seed=None):
        # Sample goal
        if self.num_goals == -1:
            goal = self.haar_states(1)
        elif self.num_goals == 1:
            goal = self.goals
        else:
            sample = np.random.randint(self.num_goals)
            goal = self.goals[sample]

        # Deal with inverse case
        if self.inverse_circuit == True:
            self.start_state = goal
            self.desired_goal = self.kets['|'+'0'*self.N+'>']
        else: 
            self.start_state = self.kets['|'+'0'*self.N+'>']
            self.desired_goal = goal

        # Reset circuit
        self.reset_circuit(state=self.start_state)
        self.score = 0
        self.fidelity = self.get_fidelity(self.state, self.desired_goal)

        if self.obs_type == 'Concat':
            desired_goal_np = self.desired_goal.cpu().reshape(-1,)
            self.desired_goal_np = pt.concatenate([desired_goal_np.real, desired_goal_np.imag], dim=0).numpy()

        observation = self._get_obs()
        return observation, {"Fidelity": self.fidelity.item()}

    def step(self, action):
        gate_symbol, qubit_idx, adjoint = self.explicit_actions[action]
        qgate = self.get_gate(gate_symbol=gate_symbol, qubit=qubit_idx, adjoint=adjoint)
        self.apply_gate(gate=qgate, gate_symbol=gate_symbol, qubits=qubit_idx, adjoint=adjoint)
        #self.Unitary = qgate @ self.Unitary
        #self.state = qgate @ self.state
        #self.history.loc[len(self.history)] = {'Gate':gate_symbol, 'Qubits':qubit_idx, 'Theta':0, 'Adjoint':adjoint}
        #self.counter += 1

        observation = self._get_obs()
        # Reward calculation
        #if self.obs_type == 'Box':
        reward = float(self.compute_reward(self.state, self.desired_goal, None).item())
        #elif self.obs_type == 'Dict':
        #    reward = float(self.compute_reward(observation["achieved_goal"], observation["desired_goal"], None).item())
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
            info['Goal'] = self.desired_goal.cpu().tolist() if self.inverse_circuit == False else self.start_state.cpu().tolist()
            info['Circuit'] = list(self.history.itertuples(index=False, name=None))

        return observation, reward, done, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, _info):
        #if self.obs_type == 'Dict':
        #    achieved_goal = self.np2tensor(achieved_goal)
        #    desired_goal = self.np2tensor(desired_goal)

        fidelity_ = self.get_fidelity(achieved_goal, desired_goal)
        if self.sparse_reward == True:
            reward = -(fidelity_ < self.fidelity_threshold).type(pt.float32)
        else:
            #if fidelity_ > self.fidelity_threshold:
            #    reward = np.array((self.max_steps - self.counter) + 1, dtype=np.float32)
            #else:
            #reward = pt.log(fidelity_ + 1e-20).cpu().numpy()
            reward = (fidelity_ - self.fidelity).cpu().numpy()
        self.fidelity = fidelity_
        return reward
        
    #def np2tensor(self, nparray):
    #    if nparray.ndim == 1:
    #        half = int(nparray.shape[0]/2)
    #        tensor = pt.tensor((nparray[:half] + 1j * nparray[half:]).reshape(half, 1), dtype=pt.complex64, device=self.device)
    #    else:
    #        # batch cases
    #        half = int(nparray.shape[1]/2)
    #        tensor = pt.tensor((nparray[:,:half] + 1j * nparray[:,half:]).reshape(nparray.shape[0],half, 1), dtype=pt.complex64, device=self.device)
    #    return tensor

    def get_fidelity(self, achieved_goal, desired_goal):
        # fidelity: |<state|goal>|^2 
        # 0 -> 1 (1 is when both states are the same)
        fidelity = pt.abs(pt.adjoint(achieved_goal) @ desired_goal)**2
        return fidelity.squeeze(dim=-1).cpu()

    def render(self):
        self.draw_circuit()

    def close(self):
        pass