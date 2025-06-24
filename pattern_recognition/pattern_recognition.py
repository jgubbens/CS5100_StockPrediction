import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm import tqdm

SEQUENCE_LENGTH = 300

def parse_ohlcv(ohlcv_data: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame(ohlcv_data)
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)
    return df

def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    return (df - df.mean()) / df.std()

def encode_horizontal_lines(df: pd.DataFrame, lines: List[Dict], num_levels=100) -> int:
    price_range = df[['low', 'high']].agg(['min', 'max']).values
    min_price, max_price = price_range[0][0], price_range[1][1]
    levels = np.linspace(min_price, max_price, num_levels)

    if not lines:
        return -1 # No label present

    target_line = lines[0]['price'] # Use first line
    idx = (np.abs(levels - target_line)).argmin()
    return idx

def encode_ray_lines(df: pd.DataFrame, rays: List[Dict]) -> np.ndarray:
    price_targets = np.full(len(df), np.nan)
    for ray in rays:
        start_time = pd.to_datetime(ray['start_date'])
        end_time = pd.to_datetime(ray['end_date'])
        start_price = ray['start_price']
        end_price = ray['end_price']

        mask = (df.index >= start_time) & (df.index <= end_time)
        time_indices = df.index[mask]

        if len(time_indices) == 0:
            continue

        t0, t1 = time_indices[0], time_indices[-1]
        delta_seconds = (t1 - t0).total_seconds()
        for t in time_indices:
            ratio = (t - t0).total_seconds() / delta_seconds if delta_seconds > 0 else 0
            price_targets[df.index.get_loc(t)] = start_price + (end_price - start_price) * ratio

    return price_targets

# Data Definition
class SupportResistanceDataset(Dataset):
    def __init__(self, json_data: Dict, sequence_length: int = 300):
        df = parse_ohlcv(json_data['ohlcv_data'])
        df = normalize_ohlcv(df)
        self.features = df[['open', 'high', 'low', 'close', 'volume']].values.T

        h_idx = encode_horizontal_lines(df, json_data['labels']['horizontal_lines'])
        self.h_class = h_idx  # integer label

        self.ray_labels = encode_ray_lines(df, json_data['labels']['ray_lines'])
        self.sequence_length = sequence_length
        self.indices = list(range(len(df) - sequence_length + 1))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start = self.indices[idx]
        end = start + self.sequence_length
        x = self.features[:, start:end]
        r = self.ray_labels[start:end]
        r = np.nan_to_num(r, nan=0.0)
        return torch.tensor(x, dtype=torch.float32), torch.tensor(self.h_class, dtype=torch.long), torch.tensor(r, dtype=torch.float32)

# Model Definition
class SupportResistanceCNN(nn.Module):
    def __init__(self, sequence_length: int = 300):
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

# Data Loader
def load_datasets_from_directory(directory: str, sequence_length: int = 300) -> ConcatDataset:
    datasets = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            file_path = os.path.join(directory, filename)
            print(f"Loading {file_path}")
            with open(file_path, "r") as f:
                try:
                    json_data = json.load(f)
                    ds = SupportResistanceDataset(json_data, sequence_length=sequence_length)
                    if ds.h_class != -1 and len(ds) > 0:
                        datasets.append(ds)
                        print(f"Loaded {filename} with {len(ds)} samples")
                    else:
                        print(f"Skipped {filename} (no label or empty)")
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")
    print(f"Total files loaded: {len(datasets)}")
    return ConcatDataset(datasets)

# Training Loop
def train_model(dataset: Dataset, epochs=10, batch_size=32):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    if isinstance(dataset, ConcatDataset):
        length = dataset.datasets[0].sequence_length
    else:
        length = dataset.dequence_length
    model = SupportResistanceCNN(sequence_length=length)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion_h = nn.CrossEntropyLoss()
    criterion_r = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        print(f"\nEpoch {epoch+1}/{epochs}")
        progress_bar = tqdm(dataloader, desc=f"Training", unit="batch")

        for x, h, r in progress_bar:
            optimizer.zero_grad()
            pred_h, pred_r = model(x)
            loss_h = criterion_h(pred_h, h)
            loss_r = criterion_r(pred_r, r)
            loss = loss_h + loss_r
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Avg Loss: {avg_loss:.4f}")

    return model

if __name__ == "__main__":
    directory = "data/train"
    print(f"Reading all JSON files from: {directory}")
    dataset = load_datasets_from_directory(directory, sequence_length=SEQUENCE_LENGTH)
    print(f"Total training samples: {len(dataset)}")
    model = train_model(dataset, epochs=10, batch_size=16)
    torch.save(model.state_dict(), f"support_resistance_cnn_{SEQUENCE_LENGTH}candlesticks.pt")
    print("\nTraining complete. Model saved.")
