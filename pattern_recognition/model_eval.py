import torch
import numpy as np
import json
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pattern_recognition import SupportResistanceCNN, load_datasets_from_directory, parse_ohlcv, normalize_ohlcv

SEQUENCE_LENGTH = 300

def evaluate_model(model, dataset):
    dataloader = DataLoader(dataset, batch_size=32)
    model.eval()
    all_preds_h, all_labels_h = [], []
    all_preds_r, all_labels_r = [], []

    with torch.no_grad():
        for x, h, r in dataloader:
            pred_h, pred_r = model(x)

            # Convert logits to binary predictions (threshold=0.5)
            pred_h_binary = (torch.sigmoid(pred_h) > 0.5).int()
            h_binary = h.int()

            all_preds_h.append(pred_h_binary)
            all_labels_h.append(h_binary)
            all_preds_r.append(pred_r)
            all_labels_r.append(r)

    # Concatenate all batches
    pred_h = torch.cat(all_preds_h).cpu().numpy()
    true_h = torch.cat(all_labels_h).cpu().numpy()
    pred_r = torch.cat(all_preds_r).cpu().numpy()
    true_r = torch.cat(all_labels_r).cpu().numpy()

    # Compute classification metrics
    acc = accuracy_score(true_h.flatten(), pred_h.flatten())
    prec = precision_score(true_h.flatten(), pred_h.flatten(), zero_division=0)
    rec = recall_score(true_h.flatten(), pred_h.flatten(), zero_division=0)
    f1 = f1_score(true_h.flatten(), pred_h.flatten(), zero_division=0)

    # Compute regression metrics
    mse = np.mean((pred_r - true_r) ** 2)
    mae = np.mean(np.abs(pred_r - true_r))

    print("Horizontal Line Prediction Metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1 Score: {f1:.4f}")

    print("Ray Line Prediction Metrics:")
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}")


def predict_single_example(model, json_path):
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    df = parse_ohlcv(json_data['ohlcv_data'])
    df_norm = normalize_ohlcv(df)
    features = df_norm[['open', 'high', 'low', 'close', 'volume']].values.T
    x = features[:, -SEQUENCE_LENGTH:]
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0)  # shape (1, 5, SEQUENCE_LENGTH)

    model.eval()
    with torch.no_grad():
        pred_h, pred_r = model(x_tensor)
        probs = torch.sigmoid(pred_h).squeeze().numpy()
        ray_preds = pred_r.squeeze().numpy()

    # Recover actual price levels
    min_price = df['low'].min()
    max_price = df['high'].max()
    levels = np.linspace(min_price, max_price, 100)
    top_indices = np.argsort(probs)[-5:][::-1]

    print("\nGround Truth Horizontal Lines:")
    for line in json_data['labels'].get('horizontal_lines', []):
        line_type = line.get('type', 'unknown')
        print(f"  Price = {line['price']:.2f}, Type = {line_type}")

    print("\nGround Truth Ray Lines:")
    for line in json_data['labels'].get('ray_lines', []):
        print(f"From {line['start_date']} ({line['start_price']:.2f}) to {line['end_date']} ({line['end_price']:.2f})")

    print("Top Predicted Support/Resistance Levels:")
    for idx in top_indices:
        print(f"Level {idx}: Price = {levels[idx]:.2f}, Prob = {probs[idx]:.3f}")

    print("Ray Line Price Predictions (last 5 steps):")
    print(ray_preds[-5:])


if __name__ == "__main__":
    directory = "data/test"
    print(f"Loading dataset from {directory} with SEQUENCE_LENGTH = {SEQUENCE_LENGTH}")
    dataset = load_datasets_from_directory(directory, sequence_length=SEQUENCE_LENGTH)
    print(f"Loaded {len(dataset)} samples.")

    model = SupportResistanceCNN(sequence_length=SEQUENCE_LENGTH)
    model.load_state_dict(torch.load(f"support_resistance_cnn_{SEQUENCE_LENGTH}candlesticks.pt"))

    evaluate_model(model, dataset)

    sample_file = r"data/test/0b3167ff-9fe3-4f49-b6f7-2ee1f52e12ee.json"
    print(f"\nMaking prediction for: {sample_file}")
    predict_single_example(model, sample_file)
