import os
import json
import random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn as nn

class SupportResistanceCNN(nn.Module):
    def __init__(self, sequence_length=300):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(5, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.fc_horizontal = nn.Linear(64, 100)
        self.fc_ray = nn.Linear(64, sequence_length)

    def forward(self, x):
        x = self.conv(x)
        x = x.squeeze(-1)
        h = self.fc_horizontal(x)
        r = self.fc_ray(x)
        return h, r

def parse_ohlcv(ohlcv_data):
    df = pd.DataFrame(ohlcv_data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

def normalize_ohlcv(df):
    return (df - df.mean()) / df.std()

def get_price_levels(df, num_levels=100):
    min_price = df['low'].min()
    max_price = df['high'].max()
    return np.linspace(min_price, max_price, num_levels)

def prepare_input(df, sequence_length=300):
    df_norm = normalize_ohlcv(df)
    features = df_norm[['open', 'high', 'low', 'close', 'volume']].values.T
    x = features[:, -sequence_length:]
    return torch.tensor(x, dtype=torch.float32).unsqueeze(0)

def predict(model, x_tensor):
    model.eval()
    with torch.no_grad():
        pred_h, pred_r = model(x_tensor)
        probs = torch.sigmoid(pred_h).squeeze(0).numpy()
        rays = pred_r.squeeze(0).numpy()
    return probs, rays

def plot_prediction(df_full, levels, probs, last_close, file_name, gt_lines):
    fig, ax = plt.subplots(figsize=(14, 6))

    for i in range(len(df_full)):
        o = df_full.loc[i, 'open']
        c = df_full.loc[i, 'close']
        h = df_full.loc[i, 'high']
        l = df_full.loc[i, 'low']
        color = 'green' if c >= o else 'red'
        ax.plot([i, i], [l, h], color='black')
        ax.plot([i, i], [o, c], color=color, linewidth=2)

    supports = [(i, p) for i, p in enumerate(probs) if levels[i] < last_close]
    resistances = [(i, p) for i, p in enumerate(probs) if levels[i] >= last_close]

    top_supports = sorted(supports, key=lambda x: x[1], reverse=True)[:5]
    top_resistances = sorted(resistances, key=lambda x: x[1], reverse=True)[:5]

    # Plot predicted supports
    for idx, prob in top_supports:
        price = levels[idx]
        ax.axhline(price, linestyle='--', color='blue', label=f"Predicted Support {idx}: {price:.2f} (p={prob:.2f})")

    # Plot predicted resistances
    for idx, prob in top_resistances:
        price = levels[idx]
        ax.axhline(price, linestyle='--', color='orange', label=f"Predicted Resistance {idx}: {price:.2f} (p={prob:.2f})")
    '''
    # Plot ground truth horizontal lines
    for line in gt_lines:
        price = line['price']
        ltype = line.get('type', '').lower()
        if ltype == 'support':
            color = 'blue'
        elif ltype == 'resistance':
            color = 'orange'
        else:
            color = 'gray'
        ax.axhline(price, linestyle='-', color=color, linewidth=2, label=f"GT {ltype.capitalize()} @ {price:.2f}")'''

    ax.set_xlabel("Candle Index")
    ax.set_ylabel("Price")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_dir = "data/test"
    model_input_length = 300
    model_path = f"support_resistance_cnn_{model_input_length}candlesticks.pt"
    display_candles = model_input_length

    model = SupportResistanceCNN(sequence_length=model_input_length)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

    all_files = [f for f in os.listdir(test_dir) if f.endswith('.json')]
    selected_files = random.sample(all_files, k=5)

    for file_name in selected_files:
        json_file_path = os.path.join(test_dir, file_name)
        print(f"\nProcessing file: {json_file_path}")

        with open(json_file_path, "r") as f:
            data = json.load(f)

        df = parse_ohlcv(data['ohlcv_data'])
        df_display = df.tail(display_candles).copy()
        df_display.reset_index(drop=True, inplace=True)

        levels = get_price_levels(df_display, num_levels=100)
        x_tensor = prepare_input(df_display, sequence_length=model_input_length)
        last_close = df_display['close'].iloc[-1]

        probs, rays = predict(model, x_tensor)

        print("Top Support Levels:")
        for idx, prob in sorted([(i, p) for i, p in enumerate(probs) if levels[i] < last_close], key=lambda x: x[1], reverse=True)[:5]:
            print(f"  Level {idx:>3}: Price = {levels[idx]:.2f}, Prob = {prob:.3f}, Type = support")

        print("Top Resistance Levels:")
        for idx, prob in sorted([(i, p) for i, p in enumerate(probs) if levels[i] >= last_close], key=lambda x: x[1], reverse=True)[:5]:
            print(f"  Level {idx:>3}: Price = {levels[idx]:.2f}, Prob = {prob:.3f}, Type = resistance")

        print("Ray Line Predictions (last 5 steps):", rays[-5:])
        gt_lines = data.get("labels", {}).get("horizontal_lines", [])
        plot_prediction(df_display, levels, probs, last_close, file_name, gt_lines)

