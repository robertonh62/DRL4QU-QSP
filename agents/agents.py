import numpy as np
import torch as T
from agents.NNs import DQ_nn, DuelDQ_nn
from agents.replay_memory import ReplayBuffer

class Agent():
    """
    Base class where all agents derive from

    Parameters:
    gamma:          float
        Discount factor
    lr:             float
        Learning rate for the optimizer
    n_actions:      int
        Number of actions the agent can perform
    input_dims:     int
        Size of the state vector
    mem_size:       int
        Size of the replay memory
    batch_size:     int
        Size of minibatch to train the model
    N_eps:          int
        Number of episodes the agent will train
    eps_max:        float
        Starting value of epsilon for the epsilon greedy policy
    eps_min:        float
        Final value of epsilon for the epsilon greedy policy
    replace:        tuple
        Defines how often to perform the update of the target network
        The tuple must have this format: (int, str) 
        where the first value represents the update frecuency and the second one the units of time 
        Options for units of time: 'step', 'episode'
    env_name:       str
        Name of the environment to save the file
    chkpt_dir:      str
        Folder to save the models
    """
    def __init__(self, 
                gamma: float, 
                lr: float, 
                n_actions: int, 
                input_dims: int, 
                mem_size: int, 
                batch_size: int, 
                N_eps: int, 
                eps_max: float=1.0, 
                eps_min: float=0.01,
                replace: tuple=(10, 'step'), 
                env_name: str=None, 
                chkpt_dir: str='logs'):

        self.gamma = gamma
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.N_eps = N_eps
        self.epsilon = eps_max
        self.eps_max = eps_max
        self.eps_min = eps_min
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.replace_target_cnt = replace[0]
        self.replace_target_type = replace[1]
        self.learn_step_counter = 0
        self.episode_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
    
    def choose_action(self, state, epsilon=None):
        raise NotImplementedError
    
    def store_transition(self, s, a, r, s_, done):
        self.memory.store_transition(s,a,r,s_,done)
    
    def sample_memory(self):
        state, action, reward, state_, done = self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        states_ = T.tensor(state_).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        return states, actions, rewards, states_, dones
    
    def replace_target_network(self, counter):
        if counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.eps_min + (self.eps_max - self.eps_min)*np.exp(-8*self.episode_counter/self.N_eps)
    
    def episode_end(self):
        self.episode_counter += 1
        self.decrement_epsilon()
        if self.replace_target_type == 'episode':
            self.replace_target_network(counter=self.episode_counter)

    def save_models(self):
        print('...Saving checkpoint...')
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
    
    def load_models(self):
        print('...Loading checkpoint...')
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        raise NotImplementedError

class DQN_agent(Agent):
    """
    Implementation of the Deep Q-Network (DQN) as explained in the paper https://doi.org/10.1038/nature14236
    """
    def __init__(self, *args, **kwargs):
        super(DQN_agent, self).__init__(*args, **kwargs)

        name = self.env_name + f'_DQN_g{self.gamma}_lr{self.lr}_m{self.mem_size}_Bsize{self.batch_size}_rep{self.replace_target_cnt}'

        self.q_eval = DQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name = name + '_q_eval',
                            chkpt_dir=self.chkpt_dir)
        
        self.q_next = DQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name = name + '_q_next',
                            chkpt_dir=self.chkpt_dir)
        
    def choose_action(self, state, epsilon=None):
        # Choose action acording to epsilon-greedy policy
        eps = self.epsilon if epsilon == None else epsilon
        if np.random.random() < eps:
            a = np.random.choice(self.action_space)
        else:
            s = T.tensor(state[np.newaxis, :], dtype=T.float, device=self.q_eval.device)
            A = self.q_eval.forward(s)
            a = T.argmax(A).item()
        return a
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()

        if self.replace_target_type == 'step':
            self.replace_target_network(counter=self.learn_step_counter)

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next

        loss = self.q_eval.loss(q_target,q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1


class DDQN_agent(Agent):
    """
    Implementation of the Double Deep Q-Network (DDQN) as explained in the paper https://www.arxiv.org/abs/1509.06461
    """
    def __init__(self, *args, **kwargs):
        super(DDQN_agent, self).__init__(*args, **kwargs)

        name = self.env_name + f'_DDQN_g{self.gamma}_lr{self.lr}_m{self.mem_size}_Bsize{self.batch_size}_rep{self.replace_target_cnt}'

        self.q_eval = DQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name = name + '_q_eval',
                            chkpt_dir=self.chkpt_dir)
        
        self.q_next = DQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name = name + '_q_next',
                            chkpt_dir=self.chkpt_dir)
        
    def choose_action(self, state, epsilon=None):
        # Choose action acording to epsilon-greedy policy
        eps = self.epsilon if epsilon == None else epsilon
        if np.random.random() > eps:
            s = T.tensor(state[np.newaxis, :], dtype=T.float, device=self.q_eval.device)
            A = self.q_eval.forward(s)
            a = T.argmax(A).item()
        else:
            a = np.random.choice(self.action_space)
        return a
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()

        if self.replace_target_type == 'step':
            self.replace_target_network(counter=self.learn_step_counter)

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(states_)
        q_eval = self.q_eval.forward(states_)

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma*q_next[indices, max_actions]

        loss = self.q_eval.loss(q_target,q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1


class DuelDQN_agent(Agent):
    """
    Implementation of the Dueling Deep Q-Network (DuelDQN) as explained in the paper https://arxiv.org/abs/1511.06581
    """
    def __init__(self, *args, **kwargs):
        super(DuelDQN_agent, self).__init__(*args, **kwargs)

        name = self.env_name + f'_DuelDQN_g{self.gamma}_lr{self.lr}_m{self.mem_size}_Bsize{self.batch_size}_rep{self.replace_target_cnt}'

        self.q_eval = DuelDQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name = name + '_q_eval',
                            chkpt_dir=self.chkpt_dir)
        
        self.q_next = DuelDQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name = name + '_q_next',
                            chkpt_dir=self.chkpt_dir)
        
    def choose_action(self, state, epsilon=None):
        # Choose action acording to epsilon-greedy policy
        eps = self.epsilon if epsilon == None else epsilon
        if np.random.random() > eps:
            s = T.tensor(state[np.newaxis, :], dtype=T.float, device=self.q_eval.device)
            _, A = self.q_eval.forward(s)
            a = T.argmax(A).item()
        else:
            a = np.random.choice(self.action_space)
        return a
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()

        if self.replace_target_type == 'step':
            self.replace_target_network(counter=self.learn_step_counter)

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)
        
        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next
        
        loss = self.q_eval.loss(q_target,q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1


class DuelDDQN_agent(Agent):
    """
    Implementation of the Dueling Deep Q-Network (DuelDQN) mixed with Double Deep Q-Network (DuelDQN)
    """
    def __init__(self, *args, **kwargs):
        super(DuelDDQN_agent, self).__init__(*args, **kwargs)

        name = self.env_name + f'_DuelDDQN_g{self.gamma}_lr{self.lr}_m{self.mem_size}_Bsize{self.batch_size}_rep{self.replace_target_cnt}'

        self.q_eval = DuelDQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name = name + '_q_eval',
                            chkpt_dir=self.chkpt_dir)
        
        self.q_next = DuelDQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name = name + '_q_next',
                            chkpt_dir=self.chkpt_dir)
        
    def choose_action(self, state, epsilon=None):
        # Choose action acording to epsilon-greedy policy
        eps = self.epsilon if epsilon == None else epsilon
        if np.random.random() > eps:
            s = T.tensor(state[np.newaxis, :], dtype=T.float, device=self.q_eval.device)
            _, A = self.q_eval.forward(s)
            a = T.argmax(A).item()
        else:
            a = np.random.choice(self.action_space)
        return a
    
    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()

        if self.replace_target_type == 'step':
            self.replace_target_network(counter=self.learn_step_counter)

        states, actions, rewards, states_, dones = self.sample_memory()

        indices = np.arange(self.batch_size)
        
        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(states_)
        V_s_eval, A_s_eval = self.q_eval.forward(states_)

        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))

        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        q_target = rewards + self.gamma*q_next[indices,max_actions]
        
        loss = self.q_eval.loss(q_target,q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1
