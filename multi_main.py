import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import wandb
import torchaudio

from multi_transformer import ContinuousTransformer

torch.manual_seed(10617)

# -----------------------------------------------------------
# WANDB INIT
# -----------------------------------------------------------
wandb.init(
    project="idl-final",
    name="continuous-mse-mel",
    config={
        "learning_rate": 3e-4,
        "epochs": 25,
        "batch_size": 8,
        "loss_fn": "MSE + Mel",
        "scheduler": "OneCycleLR",
    }
)

# -----------------------------------------------------------
# DATASET
# -----------------------------------------------------------
class PreprocessedTokenDataset(Dataset):
    """
    Returns:
        x: [T, 256]  (embedding)
        y: [T, 256]  (next-step embedding)
    """
    def __init__(self, data_files):
        self.samples = []
        for fp in data_files:
            data = torch.load(fp)
            for w in data["windows"]:
                self.samples.append(w)

        print(f"Loaded {len(self.samples)} windows from {len(data_files)} files.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = self.samples[idx]["x"]   # [T, 256]
        y = self.samples[idx]["y"]   # [T, 256]
        return x, y


# -----------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------
def train_on_dataset(model, train_loader, val_loader, optimizer, epochs,
                     device, scheduler=None):

    mse = nn.MSELoss()

    # Linear projection → pseudo-waveform for mel loss
    embed_to_audio = nn.Linear(model.d_out, 1).to(device)

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=24000,
        n_fft=1024,
        hop_length=256,
        n_mels=80
    ).to(device)

    train_losses, val_losses = [], []

    try:
        for epoch in range(epochs):

            # -------- TRAIN --------
            model.train()
            total_train_loss = 0

            for batch_idx, (x, y) in enumerate(train_loader):

                x = x.float().to(device)
                y = y.float().to(device)

                optimizer.zero_grad()

                pred = model(x)       # [B, T, 256]

                # Only MSE + Mel loss
                loss_total = 0.0

                # (1) MSE between embeddings
                loss_total += mse(pred, y)

                # (2) Mel Loss (pseudo audio)
                audio_pred = embed_to_audio(pred).squeeze(-1)  # [B, T]
                audio_true = embed_to_audio(y).squeeze(-1)

                mel_pred = mel(audio_pred)
                mel_true = mel(audio_true)

                loss_total += mse(mel_pred, mel_true)

                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler:
                    scheduler.step()

                total_train_loss += loss_total.item()

                if (batch_idx + 1) % 100 == 0:
                    print(f"Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss_total.item():.4f}")

            avg_train = total_train_loss / len(train_loader)
            train_losses.append(avg_train)

            # -------- VAL --------
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for x, y in val_loader:

                    x = x.float().to(device)
                    y = y.float().to(device)

                    pred = model(x)

                    loss_total = 0.0

                    loss_total += mse(pred, y)

                    audio_pred = embed_to_audio(pred).squeeze(-1)
                    audio_true = embed_to_audio(y).squeeze(-1)

                    mel_pred = mel(audio_pred)
                    mel_true = mel(audio_true)

                    loss_total += mse(mel_pred, mel_true)

                    total_val_loss += loss_total.item()

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
        print("Training interrupted — returning partial results.")
        return train_losses, val_losses


# -----------------------------------------------------------
# MAIN
# -----------------------------------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using:", device)

    data_dir = "2018_processed_four_levels"
    files = glob.glob(os.path.join(data_dir, "*.pt"))
    n = len(files)

    order = torch.randperm(n).tolist()
    train_files = [files[i] for i in order[: int(0.8*n)]]
    val_files   = [files[i] for i in order[int(0.8*n):]]

    train_set = PreprocessedTokenDataset(train_files)
    val_set   = PreprocessedTokenDataset(val_files)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=8, shuffle=False)

    print(f"Training windows: {len(train_set)}")
    print(f"Validation windows: {len(val_set)}")

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

    torch.save(model.state_dict(), "transformer.pth")
    wandb.save("transformer.pth")
    torch.save(train_losses, "train_losses.pt")
    torch.save(val_losses, "val_losses.pt")

    print("Training complete — model saved.")