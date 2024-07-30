import torch as pt
import numpy as np
import pandas as pd
import pennylane as qml
import matplotlib.pyplot as plt
from itertools import permutations
from typing import Optional

device = pt.device('cuda:0')

# Basic states and states operations

ket_0 = pt.tensor([[1.],[0 ]], dtype=pt.complex64, device=device)
ket_1 = pt.tensor([[0 ],[1.]], dtype=pt.complex64, device=device)
basis_kets = [ket_0,ket_1]

def bloch_state(theta, phi):
    state = pt.cos(theta/2)*ket_0 + pt.sin(theta/2)*pt.exp(1j*phi)*ket_1
    return state

def get_bra(state):
    return pt.adjoint(state)

def inner_prod(bra,ket):
    braket = get_bra(bra)@ket
    return braket[0][0].item()

def get_norm(state):
    norm = pt.sum(pt.abs(state)**2)
    return norm

def random_state(N):
    D = 2**N
    state = pt.rand(D ,dtype=pt.complex64)
    state = state/pt.sqrt(get_norm(state))
    return state.view(D,1)

# Create state space

def get_ket_probs(state):
    # Gets Hilbert state vector of a ket
    # from '00' gets pt.tensor([1, 0, 0, 0], dtype=complex)
    state_probs = basis_kets[int(state[-1])]
    for i in range(2,len(state)+1):
        state_probs = pt.kron(basis_kets[int(state[-i])],state_probs)
    return state_probs

def str2tensor(num_str):
    # Does this '010' -> pt.tensor([0,1,0])
    tensor = pt.zeros([len(num_str)],dtype=int, device=device)
    for i in range(len(num_str)):
        tensor[i] = int(num_str[i])
    return tensor

def get_kets_space(N):
    # kets: Dictionary with all states and their vectors in the Hilbert Space
    # basis: Tensor with all states as vectors of int ('|010>' -> pt.tensor([0,1,0]))
    nums = np.arange(2**N)
    kets = {}
    basis = pt.zeros([2**N,N],dtype=int)
    for num in nums:
        bin_num = np.binary_repr(num,width=N)
        state = '|' + bin_num + ">"
        kets[state] = get_ket_probs(bin_num)
        basis[num] = str2tensor(bin_num)
    return kets, basis

def state_str(state):
        # Print state in readable way
        D = state.shape[0]
        N = int(np.log2(D))
        state_str = ''
        kets, _ = get_kets_space(N)
        keys_str = list(kets.keys())
        for i in range(D):
            state_str += f'({state[i,0].numpy():.01f}){keys_str[i]} + '
        state_str  = state_str[:-2]

        return state_str

# One qubit gates

def universal_rot(phi, omega, theta):
    rot = pt.tensor([[pt.exp(-1j*(phi + omega)/2)*pt.cos(theta/2), -pt.exp(1j*(phi - omega)/2)*pt.sin(theta/2)],
                     [pt.exp(-1j*(phi - omega)/2)*pt.sin(theta/2),  pt.exp(1j*(phi + omega)/2)*pt.cos(theta/2)]], dtype=pt.complex64)
    return rot

id_mat = pt.tensor([[1., 0],[0, 1.]], dtype=pt.complex64, device=device)
# Pauli gates
PauliX = pt.tensor([[0, 1.],[1., 0]], dtype=pt.complex64, device=device)
PauliY = pt.tensor([[0, -1j],[1j, 0]], dtype=pt.complex64, device=device)
PauliZ = pt.tensor([[1., 0],[0, -1.]], dtype=pt.complex64, device=device)
# Clifford + T basis set
hadamard = 1.0/pt.sqrt(pt.tensor(2)) * pt.tensor([[1,1],[1,-1]], dtype=pt.complex64, device=device)
S_gate = pt.tensor([[1., 0],[0, 1j]], dtype=pt.complex64, device=device)
T_gate = pt.tensor([[1., 0],[0, pt.exp(pt.tensor(1j*pt.pi/4))]], dtype=pt.complex64, device=device)
# HRC basis set
b1 = 1.0/pt.sqrt(pt.tensor(5)) * pt.tensor([[1., 2*1j],[2*1j, 1]], dtype=pt.complex64, device=device)
b2 = 1.0/pt.sqrt(pt.tensor(5)) * pt.tensor([[1., 2],[-2, 1]], dtype=pt.complex64, device=device)
b3 = 1.0/pt.sqrt(pt.tensor(5)) * pt.tensor([[1. + 2*1j, 0.],[0, 1. - 2*1j]], dtype=pt.complex64, device=device)

# Rotations are in units of pi*radians, range: (-2,2)
def RX(theta):
    return pt.matrix_exp(-1j*pt.pi*theta*PauliX/2)

def RY(theta):
    return pt.matrix_exp(-1j*pt.pi*theta*PauliY/2)

def RZ(theta):
    return pt.matrix_exp(-1j*pt.pi*theta*PauliZ/2)

# Matrix form
def RX_m(theta):
    rot = pt.tensor([[np.cos(theta*pt.pi/2), -1j*np.sin(theta*pt.pi/2)],
                     [-1j*np.sin(theta*pt.pi/2), np.cos(theta*pt.pi/2)]], dtype=pt.complex64)
    return rot

def RY_m(theta):
    rot = pt.tensor([[np.cos(theta*pt.pi/2), -np.sin(theta*pt.pi/2)],
                     [np.sin(theta*pt.pi/2),  np.cos(theta*pt.pi/2)]], dtype=pt.complex64)
    return rot

def RZ_m(theta):
    rot = pt.tensor([[np.exp(-1j*theta*pt.pi/2), 0],
                     [0, np.exp( 1j*theta*pt.pi/2)]], dtype=pt.complex64)
    return rot


# Multiqubit gates

def get_Identity(k):
    Id = id_mat
    for i in range(0, k-1):
        Id = pt.kron(Id, id_mat)
    return Id
       
def get_chain_operator(A, L, i):
    # Get full matrix for applying Gate A on qubit i when we have L qubits
    Op = A
    if(i == 1):
        Op = pt.kron(A,get_Identity(L-1))
        return Op
    if(i == L):
        Op = pt.kron(get_Identity(L-1),A)
        return Op
    if(i>0 and i<L):
        Op = pt.kron(get_Identity(i-1), pt.kron(Op, get_Identity(L-i)))
        return Op

def get_chain_operator_multi(A, L, qubits):
    # Get full matrix for applying Gate A on qubit i->j when we have L qubits
    Op = A
    if(qubits[0] == 1 and qubits[1] == L):
        return Op
    if(qubits[0] == 1 and qubits[1] < L):
        Op = pt.kron(A, get_Identity(L-qubits[1]))
        return Op
    if(qubits[0] > 1 and qubits[1] < L):
        Op = pt.kron(get_Identity(qubits[0]-1), pt.kron(A, get_Identity(L-qubits[1])))
        return Op
    if(qubits[0] > 1 and qubits[1] == L):
        Op = pt.kron(get_Identity(qubits[0]-1),A)
        return Op

def get_chain_operators_Pauli(L):
    # Get all matrix for appliying basic gates
    Id = get_Identity(L)
    X = {}
    Y = {}
    Z = {}
    if L == 1:
        X[1] = PauliX     
        Y[1] = PauliY      
        Z[1] = PauliZ
    else:
        for qubit_i in range(1, L+1):                                    
            X[qubit_i] = get_chain_operator(PauliX, L, qubit_i)       
            Y[qubit_i] = get_chain_operator(PauliY, L, qubit_i)        
            Z[qubit_i] = get_chain_operator(PauliZ, L, qubit_i)   
    return Id, X, Y, Z

def get_chain_operators_clifford_T(L):
    S = {}
    T = {}
    Hadamard = {}
    if L == 1:
        S[1] = S_gate
        T[1] = T_gate  
        Hadamard[1] = hadamard
    else:
        for qubit_i in range(1, L+1):
            S[qubit_i] = get_chain_operator(S_gate, L, qubit_i)        
            T[qubit_i] = get_chain_operator(T_gate, L, qubit_i)       
            Hadamard[qubit_i] = get_chain_operator(hadamard, L, qubit_i)
    return S, T, Hadamard

def get_chain_operators_HRC(L):
    B1 = {}
    B2 = {}
    B3 = {}
    if L == 1:
        B1[1] = b1
        B2[1] = b2  
        B3[1] = b3
    else:
        for qubit_i in range(1, L+1):
            B1[qubit_i] = get_chain_operator(b1, L, qubit_i)        
            B2[qubit_i] = get_chain_operator(b2, L, qubit_i)       
            B3[qubit_i] = get_chain_operator(b3, L, qubit_i)
    return B1, B2, B3

# Generate states and Hamiltonians (Used for target states and unitaries) (SEND to utils file)

def get_Dickie_states(N):
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

        Dicke_states[f'|{S},{m}>'] = Hilbert_state

    return Dicke_states
 
def Ising_Hamiltonian(N,J=1,h=1):
    interaction_gates = []
    for i in range(1,N):
        if i+1 == N:
            gate = pt.t_copy(PauliX)
        else: 
            gate = pt.t_copy(id_mat)
        for j in range(N-1,0,-1):
            if (j == i or j == i+1):
                new_gate = pt.t_copy(PauliX)
            else:
                new_gate = pt.t_copy(id_mat)
            gate = pt.kron(new_gate, gate)
        interaction_gates.append(gate)

    individual_gates = []
    for i in range(1,N+1):
        if i == N:
            gate = pt.t_copy(PauliZ)
        else: 
            gate = pt.t_copy(id_mat)
        for j in range(N-1,0,-1):
            if j == i:
                new_gate = pt.t_copy(PauliZ)
            else:
                new_gate = pt.t_copy(id_mat)
            gate = pt.kron(new_gate,gate)
        individual_gates.append(gate)

    H = J*sum(interaction_gates) + h*sum(individual_gates)

    eigenvalues, eigenvectors = pt.linalg.eigh(H)

    return H, eigenvalues, eigenvectors


#########################################################################################################################
# Quantum Computer simulator ############################################################################################
class Circuit:
    def __init__(self, 
                 N: int, 
                 initial_state: Optional[pt.Tensor]=None):
        
        self.device = pt.device('cuda:0')

        self.N = N          # Number of qubits
        self.D = pow(2, N)  # Number of eigenstates
        self.kets, self.fock_basis = get_kets_space(N)
        self.qubit_range = [i for i in range(1, N+1)]
        self.qubit_combinations = self.get_qubit_combinations()

        self.pauli_gates = ['X', 'Y', 'Z']
        self.clifford_T_gates = ['S', 'T', 'H']
        self.HRC_gates = ['B1', 'B2', 'B3']
        self.rotational_gates = ['RX', 'RY', 'RZ']
        self.control_gates = ['CNOT']
        self.single_qgates = self.pauli_gates + self.clifford_T_gates + self.HRC_gates + self.rotational_gates
        self.double_qgates = self.control_gates

        # Get Sets of gates
        self.Id_gates, self.X_gates, self.Y_gates, self.Z_gates = get_chain_operators_Pauli(N)
        self.S_gates, self.T_gates, self.Hadamard_gates = get_chain_operators_clifford_T(N)
        self.B1_gates, self.B2_gates, self.B3_gates = get_chain_operators_HRC(N)

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

    def reset_circuit(self):
        self.state = self.initial_state
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
        
        self.state = full_gate@self.state

        self.history.loc[self.counter] = [gate_name, qubit, 0, False]
        self.Unitary = full_gate@self.Unitary
        self.counter += 1

    # Pauli Gates
    def X(self, qubit, adjoint=False):
        gate = self.X_gates[qubit]
        gate = pt.adjoint(gate) if adjoint==True else gate
        self.state = gate@self.state

        self.history.loc[self.counter] = ['X', qubit, 0, adjoint]
        self.Unitary = gate@self.Unitary
        self.counter += 1
    
    def Y(self, qubit, adjoint=False):
        gate = self.Y_gates[qubit]
        gate = pt.adjoint(gate) if adjoint==True else gate
        self.state = gate@self.state

        self.history.loc[self.counter] = ['Y', qubit, 0, adjoint]
        self.Unitary = gate@self.Unitary
        self.counter += 1
    
    def Z(self, qubit, adjoint=False):
        gate = self.Z_gates[qubit]
        gate = pt.adjoint(gate) if adjoint==True else gate
        self.state = gate@self.state

        self.history.loc[self.counter] = ['Z', qubit, 0, adjoint]
        self.Unitary = gate@self.Unitary
        self.counter += 1

    # Clifford + T gates
    def S(self, qubit, adjoint=False):
        gate = self.S_gates[qubit]
        gate = pt.adjoint(gate) if adjoint==True else gate
        self.state = gate@self.state

        self.history.loc[self.counter] = ['S', qubit, 0, adjoint]
        self.Unitary = gate@self.Unitary
        self.counter += 1

    def T(self, qubit, adjoint=False):
        gate = self.T_gates[qubit]
        gate = pt.adjoint(gate) if adjoint==True else gate
        self.state = gate@self.state

        self.history.loc[self.counter] = ['T', qubit, 0, adjoint]
        self.Unitary = gate@self.Unitary
        self.counter += 1

    def Hadamard(self, qubit, adjoint=False):
        gate = self.Hadamard_gates[qubit]
        gate = pt.adjoint(gate) if adjoint==True else gate
        self.state = gate@self.state

        self.history.loc[self.counter] = ['H', qubit, 0, adjoint]
        self.Unitary = gate@self.Unitary
        self.counter += 1

    # HRC Gates
    def B1(self, qubit, adjoint=False):
        gate = self.B1_gates[qubit]
        gate = pt.adjoint(gate) if adjoint==True else gate
        self.state = gate@self.state

        self.history.loc[self.counter] = ['B1', qubit, 0, adjoint]
        self.Unitary = gate@self.Unitary
        self.counter += 1

    def B2(self, qubit, adjoint=False):
        gate = self.B2_gates[qubit]
        gate = pt.adjoint(gate) if adjoint==True else gate
        self.state = gate@self.state

        self.history.loc[self.counter] = ['B2', qubit, 0, adjoint]
        self.Unitary = gate@self.Unitary
        self.counter += 1

    def B3(self, qubit, adjoint=False):
        gate = self.B3_gates[qubit]
        gate = pt.adjoint(gate) if adjoint==True else gate
        self.state = gate@self.state

        self.history.loc[self.counter] = ['B3', qubit, 0, adjoint]
        self.Unitary = gate@self.Unitary
        self.counter += 1

    # Rotations are in units of pi*radians, range: (-2,2)
    def RX(self,qubit,theta):
        rot = pt.matrix_exp(-1j*pt.pi*theta*self.X_gates[qubit]/2)
        self.state = rot@self.state

        self.history.loc[self.counter] = ['RX', qubit, theta, False]
        self.Unitary = rot@self.Unitary
        self.counter += 1

    def RY(self,qubit,theta):
        rot = pt.matrix_exp(-1j*pt.pi*theta*self.Y_gates[qubit]/2)
        self.state = rot@self.state

        self.history.loc[self.counter] = ['RY', qubit, theta, False]
        self.Unitary = rot@self.Unitary
        self.counter += 1

    def RZ(self,qubit,theta):
        rot = pt.matrix_exp(-1j*pt.pi*theta*self.Z_gates[qubit]/2)
        self.state = rot@self.state

        self.history.loc[self.counter] = ['RZ', qubit, theta, False]
        self.Unitary = rot@self.Unitary
        self.counter += 1

    # Two qubit gates
    def CNOT(self,qubits):
        # Qubits[0]: Contol qubit, Qubits[1]: Target qubit
        gate = pt.matrix_exp(pt.pi/4*(self.Id_gates - self.Z_gates[qubits[0]])@(self.Id_gates - self.X_gates[qubits[1]])*1j)
        self.state = gate@self.state

        self.history.loc[self.counter] = ['CNOT', qubits, 0, False]
        self.Unitary = gate@self.Unitary
        self.counter += 1
    
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
        
        gate = pt.adjoint(gate) if adjoint==True else gate
        return gate
    
    def draw_circuit(self):
        dev = qml.device("default.qubit", wires=tuple(np.arange(1,self.N+1)))

        def circuit():
            df = self.history
            for i in range(df.index.max()+1):
                # Pauli Gates
                if df.iloc[i].Gate == 'X':
                    if df.iloc[i].Adjoint == False:
                        qml.PauliX(wires=df.iloc[i].Qubits)
                    else:
                        qml.adjoint(qml.PauliX)(wires=df.iloc[i].Qubits)
                elif df.iloc[i].Gate == 'Y':
                    if df.iloc[i].Adjoint == False:
                        qml.PauliY(wires=df.iloc[i].Qubits)
                    else:
                        qml.adjoint(qml.PauliY)(wires=df.iloc[i].Qubits)
                elif df.iloc[i].Gate == 'Z':
                    if df.iloc[i].Adjoint == False:
                        qml.PauliZ(wires=df.iloc[i].Qubits)
                    else:
                        qml.adjoint(qml.PauliZ)(wires=df.iloc[i].Qubits)
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
                    if df.iloc[i].Adjoint == False:
                        qml.Hadamard(wires=df.iloc[i].Qubits)
                    else:
                        qml.adjoint(qml.Hadamard)(wires=df.iloc[i].Qubits)
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
                # Two qubit gates and special gates
                elif df.iloc[i].Gate == 'CNOT':
                    qml.CNOT(wires=[j for j in df.iloc[i].Qubits])
                else:
                    wires=[j for j in df.iloc[i].Qubits]
                    qml.QubitUnitary(U=get_Identity(wires[1]-wires[0]+1), wires=np.arange(wires[0],wires[1]+1), id=df.iloc[i].Gate)
            
            return qml.state()

        qnode = qml.QNode(circuit, dev)
        qml.draw_mpl(qnode, decimals=1, show_all_wires=True)()
        plt.show()
        #return np.array(qnode())
