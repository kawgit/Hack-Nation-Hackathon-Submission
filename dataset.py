import os
import shutil
import glob
import numpy as np
import torch
from torch.utils.data import IterableDataset
from huggingface_hub import snapshot_download

REPO_ID = "KernelCo/robot_control"
REPO_TYPE = "dataset"
ALLOW_PATTERNS = "data/*.npz"

class BrainWaveIntentDataset(IterableDataset):
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, "*.npz")))
        if not self.files:
            print(f"⚠️  Warning: No .npz files found in {data_dir}")

    def __iter__(self):
        for file_path in self.files:
            try:
                # allow_pickle=True is required for object arrays (strings, dicts)
                with np.load(file_path, allow_pickle=True) as data:
                    item = {}
                    
                    # Loop through ALL keys in the file (dynamic handling)
                    for key in data.files:
                        val = data[key]

                        # 1. Unpack 0-d arrays (scalars or object wrappers)
                        if val.ndim == 0:
                            val = val.item()  # converts np.array({'a':1}) -> {'a':1}

                        # 2. Convert based on Type
                        if isinstance(val, (int, float, bool, np.number)):
                            # Scalar Number -> 0-d Tensor
                            item[key] = torch.tensor(val)
                        
                        elif isinstance(val, np.ndarray) and val.dtype.kind in {'i', 'f', 'u', 'b'}:
                            # Numeric Array -> Tensor (copy required for memory safety)
                            item[key] = torch.from_numpy(val.copy())
                        
                        else:
                            # Complex types (Dict, String, List, None) -> Keep as is
                            # torch.tensor() would crash on these
                            item[key] = val
                    
                    yield item

            except Exception as e:
                print(f"❌ Error reading {file_path}: {e}")

def _download_and_flatten(local_dir):
    print(f"⬇️  Downloading {REPO_ID} to {local_dir}...")
    
    snapshot_download(
        repo_id=REPO_ID,
        repo_type=REPO_TYPE,
        local_dir=local_dir,
        allow_patterns=ALLOW_PATTERNS,
        resume_download=True
    )

    # Flatten logic
    nested_dir = os.path.join(local_dir, "data")
    if os.path.exists(nested_dir):
        for filename in os.listdir(nested_dir):
            shutil.move(os.path.join(nested_dir, filename), os.path.join(local_dir, filename))
        os.rmdir(nested_dir)

def load_dataset(local_dir="./data"):
    # Check if directory exists and has files
    if not (os.path.exists(local_dir) and len(glob.glob(os.path.join(local_dir, "*.npz"))) > 0):
        _download_and_flatten(local_dir)
    else:
        print(f"✅ BrainWaveIntentDataset found in '{local_dir}'. Skipping download.")

    return BrainWaveIntentDataset(local_dir)

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = load_dataset()
    
    print("\n🚀 Testing one batch...")
    # batch_size=1 is safest for mixed data types (dicts/strings)
    dl = DataLoader(ds, batch_size=1)
    
    for batch in dl:
        print(f"✅ Loaded Keys: {list(batch.keys())}")
        break