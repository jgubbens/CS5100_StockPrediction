import pandas as pd
from Env_4h import TradingEnv_4h
from DataExtract import fetch_binanceus_ohlcv
import numpy as np
import pickle

# Fetch data from Binance API
df_4h = fetch_binanceus_ohlcv('SOL/USDT', '4h', start_time='2024-01-01T00:00:00Z', end_time='2025-06-20T00:00:00Z')

# Initializae trading environment
env = TradingEnv_4h(df_4h)


# Discretize the state - put support, resistance, volume, and position observation into bins
def discretize_state(state, bins=[10, 10, 10]):
    support, resistance, volume, position = state

    support_bin = np.digitize(support, np.linspace(-1, 1, bins[0]))
    resistance_bin = np.digitize(resistance, np.linspace(-1, 1, bins[1]))
    volume_bin = np.digitize(volume, np.linspace(0, 1, bins[2]))

    # Map position (-1, 0, 1) to 0, 1, 2
    pos_bin = int(position + 1)

    return (support_bin, resistance_bin, volume_bin, pos_bin)


q_table = {}
q_update = {}
gamma = 0.9
episodes = 3000
epsilon = 1
decay_rate = 0.999

for episode in range(episodes):
    state = env.reset()
    state_d = discretize_state(state)
    total_reward = 0
    done = False
    # info = {'balance':100, 'position': 0}


    while not done:
        if state_d not in q_table:
            q_table[state_d] = np.zeros(env.action_space.n)
            q_update[state_d] = np.zeros(env.action_space.n)


        # Block agent from changing position prior to a close
        # This is not possible in the real market (e.g. you cannot change to a short from a call once the order has been placed)
        # if info["position"] != 0:
        #     action = 2
        #     while info["position"] != 0 :
        #         next_state, reward, done, info = env.step(action)

        # Choose a random action if random action probability is higher than epsilon        
        if np.random.rand() < epsilon: # and info["position"] == 0:
            action = env.action_space.sample()
            # next_state, reward, done, info = env.step(action)
        # Otherwise use q table
        else:
            action = np.argmax(q_table[state_d])
        
        next_state, reward, done, info = env.step(action)

        # Get the next observation and discreteize it
        next_state_d = discretize_state(next_state)

        # Put in Q-table if not present
        if next_state_d not in q_table:
            q_table[next_state_d] = np.zeros(env.action_space.n)
            q_update[next_state_d] = np.zeros(env.action_space.n)

        # Q-learning update
        eta = 1 / (q_update[state_d][action]+1)
        q_table[state_d][action] = (1 - eta) * q_table[state_d][action] + eta * (reward + gamma * np.max(q_table[next_state_d]))

        state_d = next_state_d
        total_reward += reward
    epsilon = epsilon * decay_rate

    if (episode + 1) % 100 == 0:
        trade_count = info.get('trade_count', 0)
        avg_profit = info.get('avg_profit_per_trade', 0.0)

        print(
            f"Episode {episode + 1:>4} | "
            f"Balance: {info['balance']:.2f} | "
            f"Total Reward: {total_reward:.2f} | "
            f"Average Reward: {total_reward:.2f} | "
            f"Epsilon: {epsilon:.4f} | "
            f"Trades: {trade_count:>3} | "
            f"Avg Profit/Trade: {avg_profit:.4f}"
        )


# Storing Q-table
# with open('4h_q_table.pkl', 'wb') as f:
#     pickle.dump(q_table, f)

# Storing Q-table with the lookaheads in training
with open('future_4h_q_table.pkl', 'wb') as f:
    pickle.dump(q_table, f)

