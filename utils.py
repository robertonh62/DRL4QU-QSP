import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Run agents
def fill_memory(env, agent):
    observation, _ = env.reset()
    for i in range(agent.mem_size):
        action = agent.choose_action(observation, epsilon=1)
        observation_, reward, done, trunc, info = env.step(action)
        terminal = done or trunc
        agent.store_transition(observation, action,reward, observation_,int(terminal))
        observation = observation_
        if terminal == True:
            observation, _ = env.reset()
    print('Done filling the memory')

def train_agent(env, agent, window = 100):
    results_df = pd.DataFrame(columns=['is_success', 'Fidelity', 'l', 'r', 'eps', 'Goal', 'Circuit'])
    for episode in range(1, agent.N_eps + 1):
        terminal = False
        observation, _ = env.reset()
        
        while not terminal:
            action = agent.choose_action(observation)
            observation_, reward, done, trunc, info = env.step(action)
            terminal = done or trunc
            agent.store_transition(observation,action,reward, observation_,int(terminal))
            observation = observation_
            agent.learn()
            if terminal == True:
                info['eps'] = agent.epsilon
                results_df.loc[len(results_df)] = info
                agent.episode_end()
                break
        
        if episode % window == 0:
            success = results_df['is_success'].tail(window).mean()*100
            fidelity = results_df['Fidelity'].tail(window).mean()
            lenght = results_df['l'].tail(window).mean()
            print(f'Episode {episode}: Sucess rate: {success:.01f} %, Avg fidelity: {fidelity:.03f}, Avg episode lenght: {lenght:.01f}, epsilon {agent.epsilon:.02f}')
        
    return results_df

def test_agent(env, agent, episodes=1, epsilon=0):
    results_df = pd.DataFrame(columns=['is_success', 'Fidelity', 'l', 'r', 'eps', 'Goal', 'Circuit'])
    for _ in range(episodes):
        terminal = False
        observation, _ = env.reset()
        
        while not terminal:
            action = agent.choose_action(observation, epsilon=epsilon)
            observation_, reward, done, trunc, info = env.step(action)
            terminal = done or trunc
            observation = observation_
            if terminal == True:
                info['eps'] = agent.epsilon
                results_df.loc[len(results_df)] = info
                break
    
    success = results_df['is_success'].mean()*100
    fidelity = results_df['Fidelity'].mean()
    lenght = results_df['l'].mean()

    print(f'Performance for {episodes} episodes:\n    Sucess rate: {success:.01f} % \n    Avg fidelity: {fidelity:.03f}\n    Avg episode lenght: {lenght:.01f}')
    return results_df

# Plots
def rolling_window(data, window, shadow_type='confidence'):
    rolling_mean = data.rolling(window=window).mean()
    if shadow_type == 'confidence':
        rolling_std = data.rolling(window=window).std()
        rolling_SEM = rolling_std / np.sqrt(window)
        Lower = rolling_mean - 1.96 * rolling_SEM
        Upper = rolling_mean + 1.96 * rolling_SEM
    elif shadow_type == 'min_max':
        Lower = data.rolling(window=window).min()
        Upper = data.rolling(window=window).max()
    return rolling_mean, Lower, Upper

def plot_results(df, window=100, type='state'):
    if isinstance(df, str):
        df = pd.read_csv(df, skiprows=1)

    if type=='state':
        fidelity_title = '(a) Average State Fidelity'
    elif type=='ising':
        fidelity_title = '(a) Average Spin Fidelity'
    elif type=='unitary':
        fidelity_title = '(a) Average Unitary Fidelity'

    num_episodes= len(df)
    sns.set_theme(context='paper',style='darkgrid',font_scale=1.75)

    fig, axs = plt.subplots(2, 2, figsize=(15, 8))

    rolling_mean, down_limit, upper_limit = rolling_window(df['Fidelity'], window)
    axs[0,0].plot(rolling_mean, linewidth=1.5)
    #axs[0,0].fill_between(x= df.index, y1=down_limit, y2=upper_limit, alpha=0.3)
    axs[0,0].set_ylabel(r'$F$')
    axs[0,0].set_title(fidelity_title)
    axs[0,0].set_xlim(0, num_episodes)


    rolling_mean, down_limit, upper_limit = rolling_window(df['r'], window)
    axs[0,1].plot(rolling_mean, linewidth=1.5)
    #axs[0,1].fill_between(x= df.index, y1=down_limit, y2=upper_limit, alpha=0.3)
    axs[0,1].set_ylabel(r'$G(\tau)$')
    axs[0,1].set_title('(b) Average Episode Return')
    axs[0,1].set_xlim(0, num_episodes)

    rolling_mean, down_limit, upper_limit = rolling_window(df['is_success'], window)
    axs[1,0].plot(rolling_mean*100, linewidth=1.5)
    #axs[1,0].fill_between(x= df.index, y1=down_limit*100, y2=upper_limit*100, alpha=0.3)
    axs[1,0].set_xlabel('Episodes')
    axs[1,0].set_ylabel('Solved episodes (%)')
    axs[1,0].set_title('(c) Percentage of Solved episodes')
    axs[1,0].set_xlim(0, num_episodes)
    axs[1,0].set_ylim(0, 105)

    rolling_mean, down_limit, upper_limit = rolling_window(df['l'], window)
    axs[1,1].plot(rolling_mean, linewidth=1.5)
    #axs[1,1].fill_between(x= df.index, y1=down_limit, y2=upper_limit, alpha=0.3)
    axs[1,1].set_xlabel('Episodes')
    axs[1,1].set_ylabel('Lenght sequence')
    axs[1,1].set_title('(d) Average Episode Lenght')
    axs[1,1].set_xlim(0, num_episodes)
    
    fig.tight_layout()