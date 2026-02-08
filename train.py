import torch
import torch.nn as nn
import wandb
from torch.utils.data import DataLoader
from dataset import load_dataset
from model import BrainWaveIntentModel
from checkpoint import save_checkpoint, load_checkpoint

# Config
CONFIG = {
    "lr": 1e-4,
    "batch_size": 32,
    "num_layers": 4,
    "dim": 64,
    "save_every": 500,
    "valid_every": 100,
    "checkpoint_path": "checkpoint.pt",
    "project_name": "brainwave-intent"
}

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu"))
    print("device:", device)

    # Data & Model
    ds = load_dataset()
    loader = DataLoader(ds, batch_size=CONFIG["batch_size"])
    model = BrainWaveIntentModel(num_layers=CONFIG["num_layers"], dim=CONFIG["dim"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])
    criterion = nn.CrossEntropyLoss()

    # Resume logic
    start_step, wandb_id = load_checkpoint(model, optimizer, CONFIG["checkpoint_path"])
    
    wandb.init(
        project=CONFIG["project_name"],
        id=wandb_id,
        resume="allow",
        config=CONFIG
    )

    step = start_step
    model.train()

    try:
        for batch in loader:
            step += 1
            if step <= start_step: continue

            eeg_features = batch["eeg_features"].to(device)
            moment_features = batch["moment_features"].to(device)
            labels = batch["label"].to(device)

            # Optimization
            optimizer.zero_grad()
            logits = model(eeg_features, moment_features)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            # Logging
            acc = (logits.argmax(1) == labels).float().mean()
            wandb.log({"loss": loss.item(), "train_acc": acc.item(), "step": step})

            if step % CONFIG["valid_every"] == 0:
                print(f"Step {step} | Loss: {loss.item():.4f} | Acc: {acc.item():.4f}")

            if step % CONFIG["save_every"] == 0:
                save_checkpoint(model, optimizer, step, wandb.run.id, CONFIG["checkpoint_path"])

    except KeyboardInterrupt:
        print("\n🛑 Training interrupted by user.")
        save_checkpoint(model, optimizer, step, wandb.run.id, CONFIG["checkpoint_path"])
    finally:
        wandb.finish()

if __name__ == "__main__":
    train()