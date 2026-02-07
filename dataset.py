import os
import shutil
import glob
import json
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
from huggingface_hub import snapshot_download

REPO_ID = "KernelCo/robot_control"
REPO_TYPE = "dataset"
ALLOW_PATTERNS = "data/*.npz"
SCALING_CONFIG_PATH = "scaling_config.json"
LOWER_PCT = 1.0
UPPER_PCT = 99.0
NUM_SAMPLES_FOR_STATS = 1000

class BrainWaveIntentDataset(IterableDataset):
    def __init__(self, data_dir, shuffle=True):
        self.data_dir = data_dir
        self.shuffle = shuffle
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        
        if not self.files:
            print(f"⚠️  Warning: No .npz files found in {data_dir}")
            return

        if os.path.exists(SCALING_CONFIG_PATH):
            self._load_scaling_config()
        else:
            self._generate_scaling_config()

    def _load_scaling_config(self):
        print(f"📖 Loading scaling config from {SCALING_CONFIG_PATH}")
        with open(SCALING_CONFIG_PATH, "r") as f:
            config = json.load(f)
            # Verify if the saved config matches current requested percentiles
            if config.get("percentiles") != [LOWER_PCT, UPPER_PCT]:
                print("⚠️  Percentile config mismatch. Regenerating...")
                self._generate_scaling_config()
            else:
                self.eeg_ranges = config["EEG_RANGES"]
                self.moment_ranges = config["MOMENT_RANGES"]

    def _generate_scaling_config(self):
        print(f"🧪 Calculating {LOWER_PCT}th-{UPPER_PCT}th percentiles...")
        all_eeg, all_moments = [], []
        
        # Sample raw data for stats
        subset = self.files[:NUM_SAMPLES_FOR_STATS]
        for file_path in subset:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    all_eeg.append(torch.nan_to_num(torch.tensor(data["feature_eeg"]), nan=0.0))
                    all_moments.append(torch.nan_to_num(torch.tensor(data["feature_moments"]), nan=0.0).view(-1))
            except Exception: continue

        eeg_tensor = torch.cat(all_eeg, dim=0)
        moments_tensor = torch.stack(all_moments, dim=0)

        # Calculate Percentiles
        self.eeg_ranges = []
        for i in range(eeg_tensor.shape[1]):
            low, high = np.percentile(eeg_tensor[:, i].numpy(), [LOWER_PCT, UPPER_PCT])
            self.eeg_ranges.append((float(low), float(high)))

        self.moment_ranges = []
        for i in range(moments_tensor.shape[1]):
            low, high = np.percentile(moments_tensor[:, i].numpy(), [LOWER_PCT, UPPER_PCT])
            if low == high: high += 1e-6
            self.moment_ranges.append((float(low), float(high)))

        # Save with metadata
        config_data = {
            "percentiles": [LOWER_PCT, UPPER_PCT],
            "EEG_RANGES": self.eeg_ranges,
            "MOMENT_RANGES": self.moment_ranges
        }
        with open(SCALING_CONFIG_PATH, "w") as f:
            json.dump(config_data, f, indent=4)
        print(f"✅ Created {SCALING_CONFIG_PATH}")

    def _preprocess(self, eeg_features, moment_features, label):
        eeg_features = torch.nan_to_num(torch.tensor(eeg_features), nan=0.0)
        moment_features = torch.nan_to_num(torch.tensor(moment_features).reshape(72, -1), nan=0.0)

        def normalize(features, ranges):
            for i, (low, high) in enumerate(ranges):
                if features.dim() == 1: # Moments (1D slice)
                    features[i] = (torch.clamp(features[i], low, high) - low) / (high - low)
                else: # EEG (Time x Channels)
                    features[:, i] = (torch.clamp(features[:, i], low, high) - low) / (high - low)
            return features

        eeg_features = normalize(eeg_features, self.eeg_ranges)
        
        # Moment normalization
        moment_flat = moment_features.view(-1)
        moment_flat = normalize(moment_flat, self.moment_ranges)
        moment_features = moment_flat.view(72, -1)

        return {
            "eeg_features": eeg_features,
            "moment_features": moment_features,
            "label": label
        }

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        file_list = self.files.copy()
        
        # 1. Macro-shuffling: Shuffle the file list
        if self.shuffle:
            random.shuffle(file_list)

        # 2. Multi-processing: Split files across workers
        if worker_info is not None:
            per_worker = int(np.ceil(len(file_list) / float(worker_info.num_workers)))
            iter_start = worker_info.id * per_worker
            iter_end = min(iter_start + per_worker, len(file_list))
            file_list = file_list[iter_start:iter_end]

        for file_path in file_list:
            try:
                with np.load(file_path, allow_pickle=True) as data:
                    yield self._preprocess(data["feature_eeg"], data["feature_moments"], data["label"].item())
            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")

def load_dataset(local_dir="./data", shuffle=True):
    if not (os.path.exists(local_dir) and len(glob.glob(os.path.join(local_dir, "*.npz"))) > 0):
        _download_and_flatten(local_dir)
    return BrainWaveIntentDataset(local_dir, shuffle=shuffle)

def _download_and_flatten(local_dir):
    print(f"⬇️  Downloading {REPO_ID}...")
    snapshot_download(repo_id=REPO_ID, repo_type=REPO_TYPE, local_dir=local_dir, allow_patterns=ALLOW_PATTERNS)
    nested_dir = os.path.join(local_dir, "data")
    if os.path.exists(nested_dir):
        for filename in os.listdir(nested_dir):
            shutil.move(os.path.join(nested_dir, filename), os.path.join(local_dir, filename))
        os.rmdir(nested_dir)

if __name__ == "__main__":
    ds = load_dataset()
    for batch in ds:
        print("eeg_features.shape:", batch["eeg_features"].shape)
        print("moment_features.shape:", batch["moment_features"].shape)
        print("label", batch["label"])
        break