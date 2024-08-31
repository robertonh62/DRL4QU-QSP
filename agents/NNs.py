import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
import os
import numpy as np

class DQ_nn(nn.Module):
    """
    Neural network class for the DQN agent

    Parametrs:
    lr:             float
        Learning rate for the optimizer
    n_actions:      int
        Number of actions the agent can perform
    input_dims:     int
        Size of the state vector
    name:           str
        Name of the model to save the file
    chkpt_dir:      str
        Folder to save the models
    """
    def __init__(self,
                lr: float, 
                n_actions: int, 
                input_dims: int, 
                name: str, 
                chkpt_dir: str):

        # Inherit properties of nn object to this object we are creating
        super(DQ_nn, self).__init__()
        # Where to save models
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        layer_size = 200
        # nn layers that we will use
        self.fullc1 = nn.Linear(input_dims, layer_size)
        self.fullc2 = nn.Linear(layer_size, layer_size)
        self.fullc3 = nn.Linear(layer_size, n_actions)

        # Initialize values of weights
        self.fullc1.weight.data.normal_(std=0.1)
        self.fullc1.bias.data.normal_(std=0.1)

        self.fullc2.weight.data.normal_(std=0.1)
        self.fullc2.bias.data.normal_(std=0.1)

        self.fullc3.weight.data.normal_(std=0.1)
        self.fullc3.bias.data.normal_(std=0.1)

        # set optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        # Set the device where we will run the code (GPU or CPU)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # Define forward pass of the nn
        fc1 = F.tanh(self.fullc1(state))
        fc2 = F.tanh(self.fullc2(fc1))
        actions = self.fullc3(fc2)
        return actions
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))

class DuelDQ_nn(nn.Module):
    """
    Neural network class for the DQN agent

    Parametrs:
    lr:             float
        Learning rate for the optimizer
    n_actions:      int
        Number of actions the agent can perform
    input_dims:     int
        Size of the state vector
    name:           str
        Name of the model to save the file
    chkpt_dir:      str
        Folder to save the models
    """
    def __init__(self,
                lr: float, 
                n_actions: int, 
                input_dims: int, 
                name: str, 
                chkpt_dir: str):
        # Inherit properties of nn object to this object we are creating
        super(DuelDQ_nn, self).__init__()
        # Where to save models
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

        layer_size = 200
        # nn layers that we will use
        self.fullc1 = nn.Linear(input_dims, layer_size)
        self.fullc2 = nn.Linear(layer_size, layer_size)
        # Dueling layers
        self.V = nn.Linear(layer_size,1)
        self.A = nn.Linear(layer_size,n_actions)

        # Initialize values of weights
        self.fullc1.weight.data.normal_(std=0.1)
        self.fullc1.bias.data.normal_(std=0.1)

        self.fullc2.weight.data.normal_(std=0.1)
        self.fullc2.bias.data.normal_(std=0.1)

        self.V.weight.data.normal_(std=0.1)
        self.V.bias.data.normal_(std=0.1)

        self.A.weight.data.normal_(std=0.1)
        self.A.bias.data.normal_(std=0.1)

        # set optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

        # Set the device where we will run the code (GPU or CPU)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        # Define forward pass of the nn
        fc1 = F.tanh(self.fullc1(state))
        fc2 = F.tanh(self.fullc2(fc1))
        V = self.V(fc2)
        A = self.A(fc2)
        return V, A
    
    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
