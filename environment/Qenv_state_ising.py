from gymnasium import Env, spaces
from environment.Qsim import Circuit, Ising_Hamiltonian
import numpy as np
import torch as pt
from typing import Union, Optional
from scipy.stats import unitary_group
import quspin


def rand_product_state(N, device, identical=True):
    # Generate random polarized product state, If identical=False each qubit has a unique random state
    zero_state = pt.zeros(2, dtype=pt.complex64, device=device).reshape(-1,1)
    zero_state[0] = 1

    haar_matrix = pt.tensor(unitary_group.rvs(dim=2, size=1), dtype=pt.complex64, device=device)
    random_qubit = haar_matrix @ zero_state
    state = random_qubit
    for _ in range(N-1):
        if identical == False:
            haar_matrix = pt.tensor(unitary_group.rvs(dim=2, size=1), dtype=pt.complex64, device=device)
            random_qubit = haar_matrix @ zero_state
        state = pt.kron(random_qubit, state)
    return state

def rand_reflection_state(N, device):
    # Generate random reflection-symmetric states
    state = np.random.normal(loc=0.0, scale=1.0, size=10) #+ 1.j * np.random.normal(loc=0.0, scale=1.0, size=10)
    state /= np.sqrt(np.abs(state.conj() @ state))

    state = quspin.basis.spin_basis_1d(L=N, pblock=1).project_from(state, sparse=False)
    state /= np.sqrt(np.abs(state.conj() @ state))

    state = pt.tensor(state.reshape(-1,1), device=device, dtype=pt.complex64)
    return state

def haar_states(N, num_states, device):
    # Generate random Haar states
    haar_matrices = pt.tensor(unitary_group.rvs(dim=2**N, size=num_states), dtype=pt.complex64, device=device)
    zero_state = pt.zeros(2**N, dtype=pt.complex64, device=device).reshape(-1,1)
    zero_state[0] = 1
    random_states = haar_matrices @ zero_state
    return random_states

def mix_states(N, num_states, device, prob=0):
    # Generate starting states with 25 % states being random polarized product states and
    # 75% being random reflection-symmetric states

    if num_states == 1:
        if np.random.random() < prob:
            return rand_product_state(N, device)
        else:
            return rand_reflection_state(N, device)
    
    mix_states = []
    for _ in range(num_states):
        if np.random.random() < prob:
            random_state = rand_product_state(N, device)
        else:
            random_state = rand_reflection_state(N, device)
        mix_states.append(random_state)

    return pt.stack(mix_states)

def ising_ground_states(N, num_states, device, J=-1, gx=[1.0, 1.1], gz=0):
    # Gerenate a random set of ground states of the Ising Hamiltonian

    if num_states == 1:
        Gx = np.random.uniform(gx[0], gx[1]) if isinstance(gx, list) else gx
        Gz = np.random.uniform(gz[0], gz[1]) if isinstance(gz, list) else gz
        H, eigenvalues, eigenvectors = Ising_Hamiltonian(N, J=J, gx=Gx, gz=Gz, device=device)
        return eigenvectors[0], {"J":J, "gx":Gx, "gz":Gz}
    
    ground_states = []
    J_vals = []
    gx_vals = []
    gz_vals = []
    for _ in range(num_states):
        Gx = np.random.uniform(gx[0], gx[1]) if isinstance(gx, list) else gx
        Gz = np.random.uniform(gz[0], gz[1]) if isinstance(gz, list) else gz
        H, eigenvalues, eigenvectors = Ising_Hamiltonian(N, J=J, gx=Gx, gz=Gz, device=device)
        ground_state = eigenvectors[0]
        ground_states.append(ground_state)
        J_vals.append(J)
        gx_vals.append(Gx)
        gz_vals.append(Gz)

    return pt.stack(ground_states), {"J":J_vals, "gx":gx_vals, "gz":gz_vals}


class Qenv_ising_state_gym(Env, Circuit):
    """
    Gym style environment for quantum state preparation of an Ising chain
    
    Parameters:
    N:                      int
        Number of Qubits
    goal:                   Tensor
        Target state
    starts:                  str | Tensor
        The method to generate initial states, if is a Tensor it has the initial states
        Options: 'haar', 'mix random states', 'ground states'
    num_starts:             int | None
        If starts is a str defines how many initial states to generate, if -1 it will generate a new start each time the reset function is called
    ground_states_params:   dict | None
        If starts is 'ground states' those are the values of the constants for the Ising Hamiltonian {'J': int, 'Gx': list | float 'Gz': list | float}
        For 'Gx' and 'Gz' if they are a list, the elements must be the min and max value to sample from [min, max]
    partition               float | None
        If starts is 'mix random states' it is the percentage of states that are product states, the remaiming part are symetrical states
    delta_t:                float
        time step size is: 
        + : 0.5 * np.pi/delta_t
        - : 0.5 * np.pi/(delta_t + 2.5)
    reward_function:        str
        How to calculate the reward
        Options: 'fidelity diference', 'log fidelity', 'sparse'
    max_steps:              int
        Max number of gates the circuit can apply 
    fidelity_threshold:     float  
        Threshold fidelity limit to concider a episode as succesfull
    device:                 str | None
        Device (cpu, cuda, …) on which the code should be run. If None, the code will be run on the GPU if possible 
    """
    def __init__(self,
                 N: int,
                 goal = pt.Tensor,
                 starts: Union[pt.Tensor,str]='mix random states',
                 num_starts: Optional[int]=None,
                 ground_states_params: Optional[dict]=None,
                 partition: Optional[float] = 0.25,
                 delta_t: float=6,
                 reward_function: str='log fidelity',
                 max_steps: int=10, 
                 fidelity_threshold: float=0.99,
                 device: Optional[str]=None):
        
        super().__init__(N=N, device=device)

        self.max_steps = max_steps

        # Define action space
        dt_p = pt.tensor(0.5 * pt.pi/delta_t)
        dt_n = - pt.tensor(0.5 * pt.pi/(delta_t + 2.5))
        self.explicit_actions = []
        for gate in self.global_gates:
            for theta in [dt_p, dt_n]:
                self.explicit_actions.append((gate, theta))

        self.action_space = spaces.Discrete(len(self.explicit_actions))

        # Define Observation Space
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2**(self.N+1),), dtype=np.float32)

        # Start related stuff
        self.ground_states_params = ground_states_params
        self.partition = partition
        self.starts = starts

        if isinstance(self.starts, pt.Tensor):
            self.start_states = self.starts
            self.num_starts = self.starts.shape[0] if self.starts.ndim == 3 else 1
        else:
            self.num_starts = num_starts
            if self.num_starts != -1:
                self.start_states = self.generate_starts(self.num_starts)
        
        # Reward related stuff
        self.goal = goal
        self.spin_fidelity_threshold = fidelity_threshold
        #self.spin_fidelity_threshold = self.fidelity_threshold**(1/self.N)
        self.reward_function = reward_function

    def generate_starts(self, num_states):
        if self.starts == 'haar':
            start_states = haar_states(self.N, num_states, self.device)
        elif self.starts == 'mix random states':
            start_states = mix_states(self.N, num_states, self.device, self.partition)
        elif self.starts == 'ground states':
            start_states, ground_states_params = ising_ground_states(self.N, num_states, self.device, **self.ground_states_params)
            self.ground_states_params_picked = ground_states_params
        else:
            assert False, "Invalid options"
        return start_states

    def _get_obs(self):
        # Observation the agent will return after each step
        obs = self.state
        self.observation = obs
        obs = obs.cpu().reshape(-1,)
        obs = pt.concatenate([obs.real, obs.imag], dim=0).numpy()
        return obs
    
    def reset(self, seed=None):
        if seed != None:
            np.random.seed(seed)

        # Sample new goal from Set of Goals
        if self.num_starts == -1:
            self.start_state = self.generate_starts(num_states=1)
            if self.starts == 'ground states':
                self.ground_states_params_ = self.ground_states_params_picked
        elif self.num_starts == 1:
            self.start_state = self.start_states
            if self.starts == 'ground states':
                self.ground_states_params_ = self.ground_states_params_picked
        else:
            sample = np.random.randint(self.num_starts)
            self.start_state = self.start_states[sample]
            if self.starts == 'ground states':
                self.ground_states_params_ = self.ground_states_params_picked[sample]

        self.reset_circuit(state=self.start_state)
        self.fidelity = self.get_fidelity(self.state, self.goal)
        self.score = 0

        observation = self._get_obs()
        return observation, {}

    def step(self, action: int):
        gate_symbol, theta = self.explicit_actions[action]
        qgate = self.get_gate(gate_symbol=gate_symbol, qubit='all', theta=theta)
        self.apply_gate(gate=qgate, gate_symbol=gate_symbol, qubits='all', theta=theta)

        observation = self._get_obs()
        # Reward calculation
        reward = self.compute_reward(self.state, self.goal)
        self.score += reward
        # Done calculation
        done = (self.spin_fidelity > self.spin_fidelity_threshold).cpu().item()
        # Truncated and info calculation
        truncated = self.counter >= self.max_steps
        info = {'Fidelity': self.spin_fidelity.item()}
        if (done == True) or (truncated == True):
            info['is_success'] = done
            info['l'] = self.counter
            info['r'] = self.score
            info['Goal'] = self.ground_states_params_ if self.starts == 'ground states' else self.goal.cpu()
            info['Circuit'] = list(self.history.itertuples(index=False, name=None))

        return observation, reward, done, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, _info=None):
        fidelity_ = self.get_fidelity(achieved_goal, desired_goal)
        spin_fidelity = fidelity_**(1/self.N)

        if self.reward_function == 'fidelity diference':
            reward = (fidelity_ - self.fidelity).cpu().numpy()
        if self.reward_function == 'log fidelity':
            reward = (pt.log(fidelity_ + 1e-20)/self.N).cpu().item()
        if self.reward_function == 'sparse':
            reward = -(spin_fidelity < self.spin_fidelity_threshold).type(pt.float32)
        
        self.fidelity = fidelity_
        self.spin_fidelity = spin_fidelity
        return reward

    def get_fidelity(self, achieved_goal, desired_goal):
        # fidelity: |<state|goal>|^2 (0 -> 1 (1 is when both states are the same))
        fidelity = pt.abs(pt.adjoint(achieved_goal) @ desired_goal)**2
        return fidelity.squeeze(dim=-1).cpu()

    def render(self):
        self.draw_circuit()

    def close(self):
        pass