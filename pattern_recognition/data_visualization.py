import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import os

def plot_candlesticks(df, horizontal_lines=None, ray_lines=None, title="Candlestick Chart with Ground Truth Lines"):
    fig, ax = plt.subplots(figsize=(12, 6))

    # Candlesticks
    for i, (t, row) in enumerate(df.iterrows()):
        color = 'green' if row['close'] >= row['open'] else 'red'
        ax.plot([t, t], [row['low'], row['high']], color='black')  # wick
        ax.plot([t, t], [row['open'], row['close']], color=color, linewidth=4)  # body

    # Horizontal lines
    if horizontal_lines:
        for line in horizontal_lines:
            price = line['price']
            ax.axhline(price, color='blue', linestyle='--', alpha=0.6)
            ax.text(df.index[-1], price, f"{price:.2f}", va='center', ha='right', color='blue')

    # Ray lines
    if ray_lines:
        for ray in ray_lines:
            try:
                start_time = pd.to_datetime(ray['start_date'])
                end_time = pd.to_datetime(ray['end_date'])
                start_price = ray['start_price']
                end_price = ray['end_price']
                ax.plot([start_time, end_time], [start_price, end_price], color='purple', linestyle='-', alpha=0.7)
            except Exception as e:
                print(f"Error plotting ray line: {e}")

    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Price")
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d\n%H:%M'))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ---- Load and Plot ----
if __name__ == "__main__":
    json_path = "data/test/0b3167ff-9fe3-4f49-b6f7-2ee1f52e12ee.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    df = pd.DataFrame(data['ohlcv_data'])
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    horizontal_lines = data.get('labels', {}).get('horizontal_lines', [])
    ray_lines = data.get('labels', {}).get('ray_lines', [])

    plot_candlesticks(df, horizontal_lines=horizontal_lines, ray_lines=ray_lines,
                      title="Ground Truth: Horizontal + Ray Lines")
    
    # Check number of candlesticks per file
    train_dir = "data/train"
    candlestick_counts = []

    for filename in os.listdir(train_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(train_dir, filename)
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    df = pd.DataFrame(data["ohlcv_data"])
                    if "volume" in df.columns:
                        candlestick_counts.append(len(df))
            except Exception:
                continue

    total_candles = sum(candlestick_counts)
    num_files = len(candlestick_counts)
    avg_per_file = total_candles / num_files if num_files else 0

    print(f"Total candlesticks: {total_candles}")
    print(f"Files counted: {num_files}")
    print(f"Average per file: {avg_per_file:.2f}")

