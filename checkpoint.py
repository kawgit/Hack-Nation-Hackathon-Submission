import torch
import os
import wandb

def save_checkpoint(model, optimizer, step, wandb_id, checkpoint_path="checkpoint.pt"):
    state = {
        "step": step,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "wandb_id": wandb_id
    }
    torch.save(state, checkpoint_path)
    print(f"💾 Checkpoint saved at step {step}")

def load_checkpoint(model, optimizer, checkpoint_path="checkpoint.pt"):
    if not os.path.exists(checkpoint_path):
        return 0, None
    
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    state = torch.load(checkpoint_path)
    model.load_state_dict(state["model_state"])
    optimizer.load_state_dict(state["optimizer_state"])
    return state["step"], state["wandb_id"]