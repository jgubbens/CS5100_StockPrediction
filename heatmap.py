import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load your saved Q-table
with open('4h_q_table.pkl', 'rb') as f:
    q_table = pickle.load(f)

bins_support = 10
bins_resistance = 10
bins_volume = 10
pos_bin = 1  # flat position

heatmap_actions = np.full((bins_resistance + 1, bins_support + 1), -1)  # initialize with -1 (no data)

for support_bin in range(bins_support + 1):
    for resistance_bin in range(bins_resistance + 1):
        actions = []
        for volume_bin in range(bins_volume + 1):
            state_key = (support_bin, resistance_bin, volume_bin, pos_bin)
            if state_key in q_table:
                best_action = np.argmax(q_table[state_key])
                actions.append(best_action)
        if actions:
            heatmap_actions[resistance_bin, support_bin] = round(np.mean(actions))  # average preferred action (rounded)

# Plot heatmap
plt.figure(figsize=(10, 8))
cmap = sns.color_palette("viridis", as_cmap=True)
sns.heatmap(
    heatmap_actions,
    cmap=cmap,
    cbar_kws={'label': 'Preferred Action (0=Long,1=Short,2=Hold)'},
    square=True,
    xticklabels=True,
    yticklabels=True,
    linewidths=0.5,
    linecolor='gray'
)
plt.xlabel('Support Bin')
plt.ylabel('Resistance Bin')
plt.title('Preferred Action Heatmap (Position = Flat, Averaged over Volume)')
plt.show()
