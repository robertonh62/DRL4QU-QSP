import numpy as np
import torch as T
from NNs import DQ_nn, DuelDQ_nn
from replay_memory import ReplayBuffer

class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims, mem_size, batch_size, eps_min=0.01,
                 eps_dec=5e-7, replace=1000, algo=None, env_name=None, chkpt_dir='checkpoints'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.chkpt_dir = chkpt_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0

        self.memory = ReplayBuffer(mem_size, input_dims, n_actions)
    
    def choose_action(self, state):
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
    
    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()
    
    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        raise NotImplementedError

class DQN_agent(Agent):
    def __init__(self, *args, **kwargs):
        super(DQN_agent, self).__init__(*args, **kwargs)

        self.q_eval = DQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name=self.env_name+'_'+self.algo+'_q_eval',
                            chkpt_dir=self.chkpt_dir)
        
        self.q_next = DQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name=self.env_name+'_'+self.algo+'_q_next',
                            chkpt_dir=self.chkpt_dir)
        
    def choose_action(self, state):
        if np.random.random() > self.epsilon:
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

        self.replace_target_network()

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

        self.decrement_epsilon()

class DDQN_agent(Agent):
    def __init__(self, *args, **kwargs):
        super(DDQN_agent, self).__init__(*args, **kwargs)

        self.q_eval = DQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name=self.env_name+'_'+self.algo+'_q_eval',
                            chkpt_dir=self.chkpt_dir)
        
        self.q_next = DQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name=self.env_name+'_'+self.algo+'_q_next',
                            chkpt_dir=self.chkpt_dir)
        
    def choose_action(self, state):
        if np.random.random() > self.epsilon:
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

        self.replace_target_network()

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

        self.decrement_epsilon()

class DuelDQN_agent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelDQN_agent, self).__init__(*args, **kwargs)

        self.q_eval = DuelDQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name=self.env_name+'_'+self.algo+'_q_eval',
                            chkpt_dir=self.chkpt_dir)
        
        self.q_next = DuelDQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name=self.env_name+'_'+self.algo+'_q_next',
                            chkpt_dir=self.chkpt_dir)
        
    def choose_action(self, state):
        if np.random.random() > self.epsilon:
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

        self.replace_target_network()

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

        self.decrement_epsilon()

class DuelDDQN_agent(Agent):
    def __init__(self, *args, **kwargs):
        super(DuelDDQN_agent, self).__init__(*args, **kwargs)

        self.q_eval = DuelDQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name=self.env_name+'_'+self.algo+'_q_eval',
                            chkpt_dir=self.chkpt_dir)
        
        self.q_next = DuelDQ_nn(self.lr, self.n_actions, input_dims=self.input_dims, name=self.env_name+'_'+self.algo+'_q_next',
                            chkpt_dir=self.chkpt_dir)
        
    def choose_action(self, state):
        if np.random.random() > self.epsilon:
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

        self.replace_target_network()

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

        self.decrement_epsilon()