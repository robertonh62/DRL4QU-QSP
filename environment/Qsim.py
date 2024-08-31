import torch as pt
import numpy as np
import pandas as pd
import pennylane as qml
import matplotlib.pyplot as plt
from itertools import permutations
from typing import Optional

#device = pt.device('cuda:0')

# Basis states
ket_0 = pt.tensor([[1.],[0 ]], dtype=pt.complex64)
ket_1 = pt.tensor([[0 ],[1.]], dtype=pt.complex64)
basis_kets = [ket_0,ket_1]

# Create state space

def get_ket_probs(state, device='cpu'):
    # Gets Hilbert state vector of a ket
    # from '00' gets pt.tensor([1, 0, 0, 0], dtype=complex)
    state_probs = basis_kets[int(state[-1])]
    for i in range(2,len(state)+1):
        state_probs = pt.kron(basis_kets[int(state[-i])],state_probs)

    return state_probs.to(device)

def str2tensor(num_str, device='cpu'):
    # Does this '010' -> pt.tensor([0,1,0])
    tensor = pt.zeros([len(num_str)],dtype=int, device=device)
    for i in range(len(num_str)):
        tensor[i] = int(num_str[i])
    return tensor

def get_kets_space(N, device='cpu'):
    # kets: Dictionary with all states and their vectors in the Hilbert Space
    # basis: Tensor with all states as vectors of int ('|010>' -> pt.tensor([0,1,0]))
    nums = np.arange(2**N)
    kets = {}
    basis = pt.zeros([2**N,N],dtype=int)
    for num in nums:
        bin_num = np.binary_repr(num,width=N)
        state = '|' + bin_num + ">"
        kets[state] = get_ket_probs(bin_num, device=device)
        basis[num] = str2tensor(bin_num, device=device)
    return kets, basis

def state_str(state):
        # Print state in readable way
        D = state.shape[0]
        N = int(np.log2(D))
        state_str = ''
        kets, _ = get_kets_space(N)
        keys_str = list(kets.keys())
        local_state = state.cpu()
        for i in range(D):
            state_str += f'({local_state[i,0].numpy():.01f}){keys_str[i]} + '
        state_str  = state_str[:-2]

        return state_str

# One qubit gates

id_mat = pt.tensor([[1., 0],[0, 1.]], dtype=pt.complex64)
# Pauli gates
PauliX = pt.tensor([[0, 1.],[1., 0]], dtype=pt.complex64)
PauliY = pt.tensor([[0, -1j],[1j, 0]], dtype=pt.complex64)
PauliZ = pt.tensor([[1., 0],[0, -1.]], dtype=pt.complex64)
# Clifford + T basis set
hadamard = 1.0/pt.sqrt(pt.tensor(2)) * pt.tensor([[1,1],[1,-1]], dtype=pt.complex64)
S_gate = pt.tensor([[1., 0],[0, 1j]], dtype=pt.complex64)
T_gate = pt.tensor([[1., 0],[0, pt.exp(pt.tensor(1j*pt.pi/4))]], dtype=pt.complex64)
# HRC basis set
b1 = 1.0/pt.sqrt(pt.tensor(5)) * pt.tensor([[1., 2*1j],[2*1j, 1]], dtype=pt.complex64)
b2 = 1.0/pt.sqrt(pt.tensor(5)) * pt.tensor([[1., 2],[-2, 1]], dtype=pt.complex64)
b3 = 1.0/pt.sqrt(pt.tensor(5)) * pt.tensor([[1. + 2*1j, 0.],[0, 1. - 2*1j]], dtype=pt.complex64)


# Multiqubit gates

def get_Identity(k, device='cpu'):
    Id = id_mat
    for i in range(0, k-1):
        Id = pt.kron(Id, id_mat)
    return Id.to(device)
       
def get_chain_operator(A, L, i, device='cpu'):
    # Get full matrix for applying Gate A on qubit i when we have L qubits
    Op = A.to(device)
    if(i == 1):
        return pt.kron(Op, get_Identity(L-1, device))
    if(i == L):
        return pt.kron(get_Identity(L-1, device), Op)
    if(i>0 and i<L):
        return pt.kron(get_Identity(i-1, device), pt.kron(Op, get_Identity(L-i, device)))

def get_chain_operator_multi(A, L, qubits, device='cpu'):
    # Get full matrix for applying Gate A on qubit i->j when we have L qubits
    Op = A.to(device)
    if(qubits[0] == 1 and qubits[1] == L):
        return Op
    if(qubits[0] == 1 and qubits[1] < L):
        return pt.kron(Op, get_Identity(L-qubits[1], device))
    if(qubits[0] > 1 and qubits[1] < L):
        return pt.kron(get_Identity(qubits[0]-1, device), pt.kron(Op, get_Identity(L-qubits[1], device)))
    if(qubits[0] > 1 and qubits[1] == L):
        return pt.kron(get_Identity(qubits[0]-1, device),Op)

def get_chain_operators_Pauli(L, device='cpu'):
    # Get all matrix for appliying basic gates
    Id = get_Identity(L, device)
    X = {}
    Y = {}
    Z = {}
    if L == 1:
        X[1] = PauliX.to(device)     
        Y[1] = PauliY.to(device)      
        Z[1] = PauliZ.to(device)
    else:
        for qubit_i in range(1, L+1):                                    
            X[qubit_i] = get_chain_operator(PauliX, L, qubit_i, device)
            Y[qubit_i] = get_chain_operator(PauliY, L, qubit_i, device)
            Z[qubit_i] = get_chain_operator(PauliZ, L, qubit_i, device)
    return Id, X, Y, Z

def get_chain_operators_clifford_T(L, device='cpu'):
    S = {}
    T = {}
    H = {}
    if L == 1:
        S[1] = S_gate.to(device) 
        T[1] = T_gate.to(device)   
        H[1] = hadamard.to(device) 
    else:
        for qubit_i in range(1, L+1):
            S[qubit_i] = get_chain_operator(S_gate, L, qubit_i, device)        
            T[qubit_i] = get_chain_operator(T_gate, L, qubit_i, device)       
            H[qubit_i] = get_chain_operator(hadamard, L, qubit_i, device)
    return S, T, H

def get_chain_operators_HRC(L, device='cpu'):
    B1 = {}
    B2 = {}
    B3 = {}
    if L == 1:
        B1[1] = b1.to(device)
        B2[1] = b2.to(device)
        B3[1] = b3.to(device)
    else:
        for qubit_i in range(1, L+1):
            B1[qubit_i] = get_chain_operator(b1, L, qubit_i, device)        
            B2[qubit_i] = get_chain_operator(b2, L, qubit_i, device)       
            B3[qubit_i] = get_chain_operator(b3, L, qubit_i, device)
    return B1, B2, B3

# Generate states and Hamiltonians (Used for target states and unitaries)

def get_Dickie_states(N, device='cpu'):
    S = N/2
    kets, _ = get_kets_space(N)
    Dicke_states = {}

    for m in np.arange(start=-S, stop=S+1, step=1):
        n_0 = int(S-m)
        n_1 = int(S+m)

        zeros = [0]*n_0
        ones = [1]*n_1
        generator = zeros + ones

        states = list(set(permutations(generator)))
        Hilbert_state = pt.zeros_like(kets['|'+'0'*N+'>'])
        for state in states:
            ket_str = '|'
            for i in state:
                ket_str += str(i)
            ket_str += '>'
            eigenstate = kets[ket_str]
            Hilbert_state += eigenstate
        norm = pt.sum(pt.abs(Hilbert_state)**2)
        Hilbert_state = Hilbert_state/pt.sqrt(norm)

        Dicke_states[f'|{S},{m}>'] = Hilbert_state.to(device)

    return Dicke_states

def Ising_Hamiltonian(N, J=1, gx=0, gz=0, device=None):
    if device == None:
        device = pt.device('cuda:0' if pt.cuda.is_available() else 'cpu')

    interaction_gates = []
    for i in range(1,N):
        gate_i = get_chain_operator(PauliZ, N, i, device)
        gate_ip1 = get_chain_operator(PauliZ, N, i+1, device)
        gate = gate_i @ gate_ip1
        interaction_gates.append(gate)
    
    individual_gates_z = []
    for i in range(1,N+1):
        gate = get_chain_operator(PauliZ, N, i, device)
        individual_gates_z.append(gate)

    individual_gates_x = []
    for i in range(1,N+1):
        gate = get_chain_operator(PauliX, N, i, device)
        individual_gates_x.append(gate)
    
    H = J*sum(interaction_gates) - gx*sum(individual_gates_x) - gz*sum(individual_gates_z)

    eigenvalues, eigenvectors = pt.linalg.eigh(H)

    Ising_states = []
    for i in range(2**N):
        Ising_states.append(eigenvectors[:,i].view(-1,1))    
    return H, eigenvalues, Ising_states

#########################################################################################################################
# Quantum Computer simulators ############################################################################################
class Circuit:
    """
    Digital Quantum Computer Simulator

    Parameters:
    N:                  int
        Number of qubits
    initial_state:      Tensor | None
        Initial state pf the system, if None the system will be initialized to the zero state
    device:             str | None
         Device (cpu, cuda, …) on which the code should be run. If None, the code will be run on the GPU if possible 
    """

    def __init__(self, 
                 N: int, 
                 initial_state: Optional[pt.Tensor]=None,
                 device: Optional[str]=None):
        
        if device == None:
            self.device = pt.device('cuda:0' if pt.cuda.is_available() else 'cpu')
        else:
            self.device = pt.device(device)

        self.N = N
        self.D = pow(2, N)
        self.kets, self.fock_basis = get_kets_space(N, self.device)
        self.qubit_range = [i for i in range(1, N+1)]
        self.qubit_combinations = self.get_qubit_combinations()

        self.pauli_gates = ['X', 'Y', 'Z']
        self.clifford_T_gates = ['S', 'T', 'H']
        self.HRC_gates = ['B1', 'B2', 'B3']
        self.rotational_gates = ['RX', 'RY', 'RZ']
        self.control_gates = ['CNOT']
        self.global_gates = ['RX_global', 'RY_global', 'RZ_global', 'RX_global_two', 'RY_global_two', 'RZ_global_two']
        self.single_qgates = self.pauli_gates + self.clifford_T_gates + self.HRC_gates + self.rotational_gates
        self.double_qgates = self.control_gates

        # Get Sets of single gates
        self.Id_gates, self.X_gates, self.Y_gates, self.Z_gates = get_chain_operators_Pauli(N, self.device)
        self.S_gates, self.T_gates, self.Hadamard_gates = get_chain_operators_clifford_T(N, self.device)
        self.B1_gates, self.B2_gates, self.B3_gates = get_chain_operators_HRC(N, self.device)

        # Get Sets of global Pauli gates
        self.pauli_global = {}
        self.pauli_global['X'] = sum([self.X_gates[qubit] for qubit in range(1,self.N+1)])
        self.pauli_global['Y'] = sum([self.Y_gates[qubit] for qubit in range(1,self.N+1)])
        self.pauli_global['Z'] = sum([self.Z_gates[qubit] for qubit in range(1,self.N+1)])
        # Sets of global two interaction Pauli gates
        self.pauli_global_two = {}
        self.pauli_global_two['X'] = sum([self.X_gates[qubit]@self.X_gates[qubit+1] for qubit in range(1,self.N)])
        self.pauli_global_two['Y'] = sum([self.Y_gates[qubit]@self.Y_gates[qubit+1] for qubit in range(1,self.N)])
        self.pauli_global_two['Z'] = sum([self.Z_gates[qubit]@self.Z_gates[qubit+1] for qubit in range(1,self.N)])


        # Initialize Circuit
        if initial_state == None:
            self.initial_state = self.kets['|'+'0'*N+'>']
        else:
            self.initial_state = initial_state.to(device=self.device)

        self.state = self.initial_state
        self.Unitary = self.Id_gates

        self.history = pd.DataFrame(columns=['Gate', 'Qubits', 'Theta', 'Adjoint'])
        self.counter = 0
    
    # HELPER FUCTIONS
    def get_qubit_combinations(self):
        combinations = []
        for i in self.qubit_range:
            for j in self.qubit_range:
                if i != j:
                    combinations.append([i,j])
        return combinations

    def reset_circuit(self, state=None):
        self.state = self.initial_state if state==None else state
        self.Unitary = self.Id_gates
        self.history = self.history.drop(self.history.index)
        self.counter = 0
    
    def state_str(self):
        # Print state in readable way
        state_str = ''
        keys_str = list(self.kets.keys())
        local_state = self.state.cpu()
        for i in range(self.D):
            state_str += f'({local_state[i,0].numpy():.01f}){keys_str[i]} + '
        state_str  = state_str[:-2]

        return state_str
    
    def apply_gate(self, gate, gate_symbol, qubits, theta=0, adjoint=False):
        # Helper function to apply gates
        gate = pt.adjoint(gate) if adjoint==True else gate
        self.state = gate@self.state

        self.history.loc[self.counter] = [gate_symbol, qubits, theta, adjoint]
        self.Unitary = gate@self.Unitary
        self.counter += 1

    # GATES
    def Unitary_gate(self, matrix, qubit, gate_name='U'):
        # This function implements the gate implemented by the matrix in the specified qubits
        # qubit can be a list with two elements [Initial qubit, last qubit] or just an int (qubit)

        if isinstance(qubit, (list, np.ndarray)):
            # It works in more than 1 qubit
            full_gate = get_chain_operator_multi(matrix, self.N, qubit)
        else:
            # It works in only 1 qubit
            full_gate = get_chain_operator(matrix, self.N, qubit)
        
        self.apply_gate(full_gate, gate_name, qubit)

    # Pauli Gates
    def X(self, qubit):
        gate = self.X_gates[qubit]
        self.apply_gate(gate, 'X', qubit)
    
    def Y(self, qubit):
        gate = self.Y_gates[qubit]
        self.apply_gate(gate, 'Y', qubit)
    
    def Z(self, qubit):
        gate = self.Z_gates[qubit]
        self.apply_gate(gate, 'Z', qubit)

    # Clifford + T gates
    def S(self, qubit, adjoint=False):
        gate = self.S_gates[qubit]
        self.apply_gate(gate, 'S', qubit, adjoint=adjoint)

    def T(self, qubit, adjoint=False):
        gate = self.T_gates[qubit]
        self.apply_gate(gate, 'T', qubit, adjoint=adjoint)

    def H(self, qubit):
        gate = self.Hadamard_gates[qubit]
        self.apply_gate(gate, 'H', qubit)

    # HRC Gates
    def B1(self, qubit, adjoint=False):
        gate = self.B1_gates[qubit]
        self.apply_gate(gate, 'B1', qubit, adjoint=adjoint)

    def B2(self, qubit, adjoint=False):
        gate = self.B2_gates[qubit]
        self.apply_gate(gate, 'B2', qubit, adjoint=adjoint)

    def B3(self, qubit, adjoint=False):
        gate = self.B3_gates[qubit]
        self.apply_gate(gate, 'B3', qubit, adjoint=adjoint)

    # Rotations are in units of pi*radians
    def RX(self, qubit, theta):
        rot = pt.matrix_exp(-1j*pt.pi*theta*self.X_gates[qubit]/2)
        self.apply_gate(rot, 'RX', qubit, theta=theta)

    def RY(self, qubit, theta):
        rot = pt.matrix_exp(-1j*pt.pi*theta*self.Y_gates[qubit]/2)
        self.apply_gate(rot, 'RY', qubit, theta=theta)

    def RZ(self, qubit, theta):
        rot = pt.matrix_exp(-1j*pt.pi*theta*self.Z_gates[qubit]/2)
        self.apply_gate(rot, 'RZ', qubit, theta=theta)

    # Two qubit gates
    def CNOT(self, qubits):
        # Qubits[0]: Contol qubit, Qubits[1]: Target qubit
        gate = pt.matrix_exp(pt.pi/4*(self.Id_gates - self.Z_gates[qubits[0]])@(self.Id_gates - self.X_gates[qubits[1]])*1j)
        self.apply_gate(gate, 'CNOT', qubits)
    
    # Global rotation gates
    def RX_global(self, theta):
        gate = pt.matrix_exp(-1j*theta*self.pauli_global['X'])
        self.apply_gate(gate, 'RX_global', 'all', theta=theta)
    
    def RY_global(self, theta):
        gate = pt.matrix_exp(-1j*theta*self.pauli_global['Y'])
        self.apply_gate(gate, 'RY_global', 'all', theta=theta)

    def RZ_global(self, theta):
        gate = pt.matrix_exp(-1j*theta*self.pauli_global['Z'])
        self.apply_gate(gate, 'RZ_global', 'all', theta=theta)
    
    # Global two interaction rotation gates
    def RX_global_two(self, theta):
        gate = pt.matrix_exp(-1j*theta*self.pauli_global_two['X'])
        self.apply_gate(gate, 'RX_global_two', 'all', theta=theta)
    
    def RY_global_two(self, theta):
        gate = pt.matrix_exp(-1j*theta*self.pauli_global_two['Y'])
        self.apply_gate(gate, 'RY_global_two', 'all', theta=theta)

    def RZ_global_two(self, theta):
        gate = pt.matrix_exp(-1j*theta*self.pauli_global_two['Z'])
        self.apply_gate(gate, 'RZ_global_two', 'all', theta=theta)

    # Auxiliar Fuctions
    def get_gate(self, gate_symbol, qubit, theta=0, adjoint=False):
        # Only returns matrix for the gate
        gate = self.Id_gates

        # Pauli gates
        if gate_symbol == "X":
            gate = self.X_gates[qubit]
        elif gate_symbol == "Y":
            gate = self.Y_gates[qubit]
        elif gate_symbol == "Z":
            gate = self.Z_gates[qubit]
    
        # Clifford + T gates
        elif gate_symbol == "S":
            gate = self.S_gates[qubit]
        elif gate_symbol == "T":
            gate = self.T_gates[qubit]
        elif gate_symbol == "H":
            gate = self.Hadamard_gates[qubit]
        
        # HRC gates
        elif gate_symbol == "B1":
            gate = self.B1_gates[qubit]
        elif gate_symbol == "B2":
            gate = self.B2_gates[qubit]
        elif gate_symbol == "B3":
            gate = self.B3_gates[qubit]
        
        # Rotational gates
        elif gate_symbol == "RX":
            gate = pt.matrix_exp(-1j*pt.pi*theta*self.X_gates[qubit]/2)
        elif gate_symbol == "RY":
            gate = pt.matrix_exp(-1j*pt.pi*theta*self.Y_gates[qubit]/2)
        elif gate_symbol == "RZ":
            gate = pt.matrix_exp(-1j*pt.pi*theta*self.Z_gates[qubit]/2)
        
        # Multiple qubit gates
        elif gate_symbol == "CNOT":
            gate = pt.matrix_exp(pt.pi/4*(self.Id_gates - self.Z_gates[qubit[0]])@(self.Id_gates - self.X_gates[qubit[1]])*1j)
        
        # Global rotation gates
        if gate_symbol == "RX_global":
            gate = pt.matrix_exp(-1j*theta*self.pauli_global['X'])
        elif gate_symbol == "RY_global":
            gate = pt.matrix_exp(-1j*theta*self.pauli_global['Y'])
        elif gate_symbol == "RZ_global":
            gate = pt.matrix_exp(-1j*theta*self.pauli_global['Z'])
        
        # Global two interaction rotation gates
        elif gate_symbol == "RX_global_two":
            gate = pt.matrix_exp(-1j*theta*self.pauli_global_two['X'])
        elif gate_symbol == "RY_global_two":
            gate = pt.matrix_exp(-1j*theta*self.pauli_global_two['Y'])
        elif gate_symbol == "RZ_global_two":
            gate = pt.matrix_exp(-1j*theta*self.pauli_global_two['Z'])

        gate = pt.adjoint(gate) if adjoint==True else gate
        return gate
    
    def draw_circuit(self, title=None):
        dev = qml.device("default.qubit", wires=tuple(np.arange(1,self.N+1)))

        def circuit():
            df = self.history
            for i in range(df.index.max()+1):

                # Pauli Gates
                if df.iloc[i].Gate == 'X':
                    qml.PauliX(wires=df.iloc[i].Qubits)
                elif df.iloc[i].Gate == 'Y':
                    qml.PauliY(wires=df.iloc[i].Qubits)
                elif df.iloc[i].Gate == 'Z':
                    qml.PauliZ(wires=df.iloc[i].Qubits)

                # Clifford + T Gates
                elif df.iloc[i].Gate == 'S':
                    if df.iloc[i].Adjoint == False:
                        qml.S(wires=df.iloc[i].Qubits)
                    else:
                        qml.adjoint(qml.S)(wires=df.iloc[i].Qubits)
                elif df.iloc[i].Gate == 'T':
                    if df.iloc[i].Adjoint == False:
                        qml.T(wires=df.iloc[i].Qubits)
                    else:
                        qml.adjoint(qml.T)(wires=df.iloc[i].Qubits)
                elif df.iloc[i].Gate == 'H':
                    qml.Hadamard(wires=df.iloc[i].Qubits)

                # HRC Gates
                elif df.iloc[i].Gate == 'B1':
                    gate_id = 'B1' if df.iloc[i].Adjoint == False else 'B1+'
                    qml.QubitUnitary(U=b1, wires=df.iloc[i].Qubits, id=gate_id)
                elif df.iloc[i].Gate == 'B2':
                    gate_id = 'B2' if df.iloc[i].Adjoint == False else 'B2+'
                    qml.QubitUnitary(U=b2, wires=df.iloc[i].Qubits, id=gate_id)
                elif df.iloc[i].Gate == 'B3':
                    gate_id = 'B3' if df.iloc[i].Adjoint == False else 'B3+'
                    qml.QubitUnitary(U=b3, wires=df.iloc[i].Qubits, id=gate_id)

                # Rotational Gates
                elif df.iloc[i].Gate == 'RX':
                    qml.RX(df.iloc[i].Theta*pt.pi ,wires=df.iloc[i].Qubits)
                elif df.iloc[i].Gate == 'RY':
                    qml.RY(df.iloc[i].Theta*pt.pi ,wires=df.iloc[i].Qubits)
                elif df.iloc[i].Gate == 'RZ':
                    qml.RZ(df.iloc[i].Theta*pt.pi ,wires=df.iloc[i].Qubits)

                # Two qubit control gates
                elif df.iloc[i].Gate == 'CNOT':
                    qml.CNOT(wires=[j for j in df.iloc[i].Qubits])

                # Global rotation gates
                elif df.iloc[i].Gate == 'RX_global':
                    for qubit in range(1, self.N+1):
                        qml.RX(df.iloc[i].Theta ,wires=qubit)
                elif df.iloc[i].Gate == 'RY_global':
                    for qubit in range(1, self.N+1):
                        qml.RY(df.iloc[i].Theta ,wires=qubit)
                elif df.iloc[i].Gate == 'RZ_global':
                    for qubit in range(1, self.N+1):
                        qml.RZ(df.iloc[i].Theta ,wires=qubit)

                # Global two interaction rotation gates
                elif df.iloc[i].Gate == 'RX_global_two':
                    for qubit in range(1, self.N):
                        qml.QubitUnitary(U=get_Identity(2), wires=[qubit,qubit+1], id=f"Rxx({df.iloc[i].Theta})")
                elif df.iloc[i].Gate == 'RY_global_two':
                    for qubit in range(1, self.N):
                        qml.QubitUnitary(U=get_Identity(2), wires=[qubit,qubit+1], id=f"Ryy({df.iloc[i].Theta})")
                elif df.iloc[i].Gate == 'RZ_global_two':
                    for qubit in range(1, self.N):
                        qml.QubitUnitary(U=get_Identity(2), wires=[qubit,qubit+1], id=f"Rzz({df.iloc[i].Theta})")

                # else
                else:
                    wires=[j for j in df.iloc[i].Qubits]
                    qml.QubitUnitary(U=get_Identity(wires[1]-wires[0]+1), wires=np.arange(wires[0],wires[1]+1), id=df.iloc[i].Gate)
            
            return qml.state()

        qnode = qml.QNode(circuit, dev)
        fig, ax = qml.draw_mpl(qnode, decimals=2, show_all_wires=True)()
        if title != None:
            plt.title(title,fontsize=20, pad=0)
        plt.show()


class Circuit_tevolve:
    def __init__(self, 
                 N: int, 
                 initial_state: Optional[pt.Tensor]=None):
        
        self.device = pt.device('cuda:0')

        self.N = N          # Number of qubits
        self.D = pow(2, N)  # Number of eigenstates
        self.kets, self.fock_basis = get_kets_space(N)
        self.qubit_range = [i for i in range(1, N+1)]
        self.qubit_combinations = self.get_qubit_combinations()

        self.gates_symbols = ['RX_global', 'RY_global', 'RZ_global', 'RX_two', 'RY_two', 'RZ_two']

        # Get Sets of single gates
        self.Id_gates, self.X_gates, self.Y_gates, self.Z_gates = get_chain_operators_Pauli(N)
        # Sets of global gates
        self.pauli_global = {}
        self.pauli_global['X'] = sum([self.X_gates[qubit] for qubit in range(1,self.N+1)])
        self.pauli_global['Y'] = sum([self.Y_gates[qubit] for qubit in range(1,self.N+1)])
        self.pauli_global['Z'] = sum([self.Z_gates[qubit] for qubit in range(1,self.N+1)])
        # Sets of global two interaction gates
        self.pauli_global_two = {}
        self.pauli_global_two['X'] = sum([self.X_gates[qubit]@self.X_gates[qubit+1] for qubit in range(1,self.N)])
        self.pauli_global_two['Y'] = sum([self.Y_gates[qubit]@self.Y_gates[qubit+1] for qubit in range(1,self.N)])
        self.pauli_global_two['Z'] = sum([self.Z_gates[qubit]@self.Z_gates[qubit+1] for qubit in range(1,self.N)])

        # Initialize Circuit
        if initial_state == None:
            self.initial_state = self.kets['|'+'0'*N+'>']
        else:
            self.initial_state = initial_state.to(device=self.device)

        self.state = self.initial_state
        self.Unitary = self.Id_gates

        self.history = pd.DataFrame(columns=['Gate', 'Theta'])
        self.counter = 0

    def get_qubit_combinations(self):
        pairs = []
        for i in range(1, self.N):
            pairs.append([i,i+1])
        return pairs

    def reset_circuit(self, state=None):
        self.state = self.initial_state if state==None else state
        self.Unitary = self.Id_gates
        self.history = self.history.drop(self.history.index)
        self.counter = 0
    
    def state_str(self):
        # Print state in readable way
        state_str = ''
        keys_str = list(self.kets.keys())
        local_state = self.state.cpu()
        for i in range(self.D):
            state_str += f'({local_state[i,0].numpy():.01f}){keys_str[i]} + '
        state_str  = state_str[:-2]

        return state_str
    
    def apply_gate(self, gate, gate_symbol, theta=0):
        # Helper function to apply gates
        self.state = gate@self.state

        self.history.loc[self.counter] = [gate_symbol, theta]
        self.Unitary = gate@self.Unitary
        self.counter += 1

    # Gates
    def RX_global(self,theta):
        gate = pt.matrix_exp(-1j*theta*self.pauli_global['X'])
        self.apply_gate(gate, 'RX_global', theta=theta)
    
    def RY_global(self,theta):
        gate = pt.matrix_exp(-1j*theta*self.pauli_global['Y'])
        self.apply_gate(gate, 'RY_global', theta=theta)

    def RZ_global(self,theta):
        gate = pt.matrix_exp(-1j*theta*self.pauli_global['Z'])
        self.apply_gate(gate, 'RZ_global', theta=theta)
    
    def RX_global_two(self,theta):
        gate = pt.matrix_exp(-1j*theta*self.pauli_global_two['X'])
        self.apply_gate(gate, 'RX_global_two', theta=theta)
    
    def RY_global_two(self,theta):
        gate = pt.matrix_exp(-1j*theta*self.pauli_global_two['Y'])
        self.apply_gate(gate, 'RY_global_two', theta=theta)

    def RZ_global_two(self,theta):
        gate = pt.matrix_exp(-1j*theta*self.pauli_global_two['Z'])
        self.apply_gate(gate, 'RZ_global_two', theta=theta)

    # Auxiliar Fuctions
    def get_gate(self, gate_symbol, theta=0):
        # Only returns matrix for the gate

        if gate_symbol == "RX_global":
            return pt.matrix_exp(-1j*theta*self.pauli_global['X'])
        elif gate_symbol == "RY_global":
            return pt.matrix_exp(-1j*theta*self.pauli_global['Y'])
        elif gate_symbol == "RZ_global":
            return pt.matrix_exp(-1j*theta*self.pauli_global['Z'])
        
        elif gate_symbol == "RX_global_two":
            return pt.matrix_exp(-1j*theta*self.pauli_global_two['X'])
        elif gate_symbol == "RY_global_two":
            return pt.matrix_exp(-1j*theta*self.pauli_global_two['Y'])
        elif gate_symbol == "RZ_global_two":
            return pt.matrix_exp(-1j*theta*self.pauli_global_two['Z'])

        return self.Id_gates

    def draw_circuit(self):
        dev = qml.device("default.qubit", wires=tuple(np.arange(1,self.N+1)))

        def circuit():
            df = self.history
            for i in range(df.index.max()+1):
                wires=[j for j in df.iloc[i].Qubits]
                qml.QubitUnitary(U=get_Identity(wires[1]-wires[0]+1), wires=np.arange(wires[0],wires[1]+1), id=df.iloc[i].Gate)
            return qml.state()

        qnode = qml.QNode(circuit, dev)
        qml.draw_mpl(qnode, decimals=1, show_all_wires=True, style='sketch')()
        plt.show()
        #return np.array(qnode())