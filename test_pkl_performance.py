from Env_5m import TradingEnv_5m 
from DataExtract import fetch_binanceus_ohlcv
import numpy as np
import pickle
import matplotlib.pyplot as plt
import os


with open("future_4h_q_table.pkl", "rb") as f:
    q_table_4h = pickle.load(f)

df_4h = fetch_binanceus_ohlcv('SOL/USDT', '4h', start_time='2025-06-25T00:00:00Z', end_time='2025-06-26T00:00:00Z')
df_4h['timestamp'] = df_4h.index
df_4h.reset_index(drop=True, inplace=True)

df_5m = fetch_binanceus_ohlcv('BTC/USDT', '5m', start_time='2025-06-25T00:00:00Z', end_time='2025-06-26T00:00:00Z')
df_5m['timestamp'] = df_5m.index
df_5m.reset_index(drop=True, inplace=True)

env = TradingEnv_5m(df_5m, df_4h, q_table_4h)

# def discretize_state(state, bins=[10, 10, 10, 4]):
#     return tuple(np.digitize(s, np.linspace(-1, 1, b)) for s, b in zip(state, bins))

def discretize_state(state, bins_continuous=[10, 10, 10]):
    # Unpack 5m continuous features
    support_5m, resistance_5m, volume_5m = state[:3]
    # 4h action one-hot vector
    action_4h_onehot = state[3:]

    # Discretize continuous values to bins 0..bins_continuous[i]
    support_bin = np.digitize(support_5m, np.linspace(-1, 1, bins_continuous[0]))
    resistance_bin = np.digitize(resistance_5m, np.linspace(-1, 1, bins_continuous[1]))
    volume_bin = np.digitize(volume_5m, np.linspace(0, 1, bins_continuous[2]))

    # Convert 4h one-hot vector to discrete action (0,1,2,3)
    action_4h_bin = np.argmax(action_4h_onehot)  # value from 0 to 3

    # Return combined discrete state tuple
    return (support_bin, resistance_bin, volume_bin, action_4h_bin)

q_table = {}
q_update = {}
gamma = 0.95
episodes = 1
reward_log = []
epsilon = 0 
decay_rate = 0.99

if os.path.exists("future_4h_q_table.pkl"):
    with open("future_4h_q_table.pkl", "rb") as f:
        q_table = pickle.load(f)
        # Also initialize q_update keys
        for state in q_table:
            q_update[state] = np.zeros(len(q_table[state]))


for episode in range(episodes):
    state = env.reset()
    state_d = discretize_state(state)
    total_reward = 0
    done = False

    while not done:
        if state_d not in q_table:
            q_table[state_d] = np.zeros(env.action_space.n)
            q_update[state_d] = np.zeros(env.action_space.n)

        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state_d])
            # if action != 0:
                # print(action)
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

    reward_log.append(total_reward)
    true_profit = info['balance'] - 100.0

    if (episode + 1) % 1 == 0:
        recent_rewards = reward_log[-10:]
        reward_std = np.std(recent_rewards)
        print(f"Episode {episode+1}, Final Balance: {info['balance']:.2f}, Total Reward: {total_reward:.2f}, True PnL: {true_profit:.2f}, StdDev (last 10): {reward_std:.2f}, Epsilon: {epsilon:.4f}")

# plot the reward scatter graph
plt.figure(figsize=(10, 5))
plt.plot(reward_log, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("5m Q-learning Training Reward Progress")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("reward_progress.png")
plt.show()
