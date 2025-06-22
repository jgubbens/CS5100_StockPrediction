import pandas as pd
from Env_4h import TradingEnv_4h
from DataExtract import fetch_binanceus_ohlcv
import numpy as np
import pickle

df_4h = fetch_binanceus_ohlcv('SOL/USDT', '4h', start_time='2024-01-18T00:00:00Z', end_time='2025-06-20T00:00:00Z')

env = TradingEnv_4h(df_4h)

def discretize_state(state, bins=[10, 10, 10, 3]):
    return tuple(np.digitize(s, np.linspace(-1, 1, b)) for s, b in zip(state, bins))

q_table = {}
q_update = {}
gamma = 0.95 
episodes = 100

for episode in range(episodes):
    epsilon = 0.1 
    decay_rate = 0.9
    state = env.reset()
    state_d = discretize_state(state)
    total_reward = 0
    done = False

    while not done:
        if state_d not in q_table:
            q_table[state_d] = np.zeros(env.action_space.n)
            q_update[state_d] = np.zeros(env.action_space.n)


        # epsilon-greedy
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_d])

        next_state, reward, done, info = env.step(action)
        next_state_d = discretize_state(next_state)

        if next_state_d not in q_table:
            q_table[next_state_d] = np.zeros(env.action_space.n)
            q_update[next_state_d] = np.zeros(env.action_space.n)

        # Q-learning update
        eta = 1 / (q_update[state_d][action]+1)
        q_table[state_d][action] = (1 - eta) * q_table[state_d][action] + eta * (reward + gamma * np.max(q_table[next_state_d]))

        state_d = next_state_d
        total_reward += reward
        epsilon = epsilon * decay_rate

    print(f"Episode {episode+1}, Final Balance: {info['balance']:.2f}, Total Reward: {total_reward:.2f}")


with open('4h_q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f)

