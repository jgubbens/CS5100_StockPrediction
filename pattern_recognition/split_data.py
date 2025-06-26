import os
import shutil
import random

def split_json_data(data_dir="data", train_dir="train", test_dir="test", test_ratio=0.2):
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    all_files = [f for f in os.listdir(data_dir) if f.endswith(".json")]
    random.shuffle(all_files)

    test_count = int(len(all_files) * test_ratio)
    test_files = all_files[:test_count]
    train_files = all_files[test_count:]

    for f in train_files:
        shutil.copy(os.path.join(data_dir, f), os.path.join(train_dir, f))
    for f in test_files:
        shutil.copy(os.path.join(data_dir, f), os.path.join(test_dir, f))

    print(f"{len(train_files)} files copied to {train_dir}")
    print(f"{len(test_files)} files copied to {test_dir}")

if __name__ == "__main__":
    split_json_data(data_dir="data", train_dir="train", test_dir="test", test_ratio=0.2)
