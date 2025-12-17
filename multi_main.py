import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb

from multi_transformer import ContinuousTransformer

torch.manual_seed(10617)

# -----------------------------------------------------------
# WANDB INIT
# -----------------------------------------------------------
wandb.init(
    project="idl-final",
    name="continuous-mse-only",
    config={
        "learning_rate": 3e-4,
        "epochs": 25,
        "batch_size": 8,
        "loss_fn": "MSE",
        "scheduler": "OneCycleLR",
    }
)

# -----------------------------------------------------------
# DATASET
# -----------------------------------------------------------
class PreprocessedTokenDataset(Dataset):
    """
    Lazy dataset for loading embedding windows from .pt files.

    Each window contains:
        x: [T, 256]
        y: [T, 256]
    """
    def __init__(self, data_files):
        self.data_files = data_files
        self.index = []   # list of (file_idx, window_idx)

        # Build lightweight index
        for fi, fp in enumerate(self.data_files):
            data = torch.load(fp, map_location="cpu")
            num_windows = len(data["windows"])
            for wi in range(num_windows):
                self.index.append((fi, wi))

        print(f"Indexed {len(self.index)} windows from {len(self.data_files)} files.")

        # File cache
        self._cache_idx = None
        self._cache_data = None

    def __len__(self):
        return len(self.index)

    def _load_file(self, file_idx):
        if file_idx != self._cache_idx:
            fp = self.data_files[file_idx]
            self._cache_data = torch.load(fp, map_location="cpu")
            self._cache_idx = file_idx
        return self._cache_data

    def __getitem__(self, idx):
        file_idx, window_idx = self.index[idx]
        data = self._load_file(file_idx)
        window = data["windows"][window_idx]
        return window["x"], window["y"]


# -----------------------------------------------------------
# TRAINING LOOP (MSE ONLY)
# -----------------------------------------------------------
def train_on_dataset(model, train_loader, val_loader, optimizer, epochs, device, scheduler=None):
    mse = nn.MSELoss()

    train_losses = []
    val_losses = []

    try:
        for epoch in range(epochs):

            # ---------------- TRAIN ----------------
            model.train()
            total_train_loss = 0

            for batch_idx, (x, y) in enumerate(train_loader):

                x = x.float().to(device)
                y = y.float().to(device)

                optimizer.zero_grad()

                pred = model(x)
                loss = mse(pred, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()

                total_train_loss += loss.item()

                if (batch_idx + 1) % 100 == 0:
                    print(f"Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

            avg_train = total_train_loss / len(train_loader)
            train_losses.append(avg_train)

            # ---------------- VALIDATION ----------------
            model.eval()
            total_val_loss = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.float().to(device)
                    y = y.float().to(device)
                    pred = model(x)
                    loss = mse(pred, y)
                    total_val_loss += loss.item()

            avg_val = total_val_loss / len(val_loader)
            val_losses.append(avg_val)

            print(f"Epoch {epoch+1}/{epochs} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")

            wandb.log({
                "train_loss": avg_train,
                "val_loss": avg_val,
                "epoch": epoch+1
            })

        return train_losses, val_losses

    except KeyboardInterrupt:
        print("\nTRAINING INTERRUPTED — saving partial model...")

        # Save partially trained model
        torch.save(model.state_dict(), "transformer_partial.pth")
        wandb.save("transformer_partial.pth")

        torch.save(train_losses, "train_losses_partial.pt")
        torch.save(val_losses, "val_losses_partial.pt")

        print("Partial model saved as transformer_partial.pth")
        print("Partial losses saved.")

        return train_losses, val_losses


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    # ---------------- LOAD DATA ----------------
    data_dir = "2018_embedded"
    files = glob.glob(os.path.join(data_dir, "*.pt"))
    n = len(files)

    # Random train / val split
    order = torch.randperm(n).tolist()
    train_files = [files[i] for i in order[: int(0.8 * n)]]
    val_files   = [files[i] for i in order[int(0.8 * n):]]

    train_set = PreprocessedTokenDataset(train_files)
    val_set   = PreprocessedTokenDataset(val_files)

    train_loader = DataLoader(
        train_set,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    print(f"Training windows:   {len(train_set)}")
    print(f"Validation windows: {len(val_set)}")

    # ---------------- MODEL ----------------
    model = ContinuousTransformer(
        d_in=256,
        d_model=512,
        d_out=256,
        num_heads=8,
        num_layers=6,
        ff_dim=2048,
        max_seq_len=512
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=3e-4)

    epochs = 25

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=6e-4,
        steps_per_epoch=len(train_loader),
        epochs=epochs,
        pct_start=0.1
    )

    # ---------------- START TRAINING ----------------
    print("==== TRAINING START ====")

    train_losses, val_losses = train_on_dataset(
        model,
        train_loader,
        val_loader,
        optimizer,
        epochs=epochs,
        device=device,
        scheduler=scheduler
    )

    # ---------------- SAVE ----------------
    torch.save(model.state_dict(), "transformer.pth")
    wandb.save("transformer.pth")

    torch.save(train_losses, "train_losses.pt")
    torch.save(val_losses, "val_losses.pt")

    print("Training complete — model saved.")