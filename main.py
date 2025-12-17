import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from transformer import Transformer
from torch.utils.data import random_split
import wandb

torch.manual_seed(10617)

wandb.init(
    project="idl-final",         
    name="final-4-quantizers", 
    config={
        "learning_rate": 3e-4,
        "epochs": 20,
        "batch_size": 8,
        "optimizer": "Adam",
        "loss_fn": "CrossEntropy",
    }
)

# Dataloader
class PreprocessedTokenDataset(Dataset):
    def __init__(self, data_files):
        self.data_files = data_files
        self.samples = []

        for file_path in self.data_files:
            data = torch.load(file_path)
            for window in data['windows']:
                self.samples.append(window)

        print(f"Loaded {len(self.samples)} windows from {len(self.data_files)} song files")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        x = sample['x']
        y = sample['y']
        return x, y


# Training and Validation Loop
def train_on_dataset(model, train_loader, val_loader, optimizer, criterion, epochs, device="cpu"):
    model.to(device)

    # Track losses
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        # Training
        model.train()
        total_train_loss = 0.0

        for batch_idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item()

            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} | Loss: {loss.item():.4f}")

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        wandb.log({
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "epoch": epoch + 1
        })

    
    return train_losses, val_losses
    



if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # Path to WAV
    data_dir = "2018_processed"  # Directory with .pt files

    # Model setup
    model = Transformer()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    # Build dataset by songs instead of samples
    data_files = glob.glob(os.path.join(data_dir, "*.pt"))
    num_files = len(data_files)

    # Determine train-val file-level split
    train_size = int(0.8 * num_files)
    val_size = num_files - train_size

    # Shuffle and split file paths
    file_paths = torch.randperm(num_files).tolist()
    train_files = [data_files[i] for i in file_paths[:train_size]]
    val_files = [data_files[i] for i in file_paths[train_size:]]

    # Create dataset instances
    train_dataset = PreprocessedTokenDataset(train_files)
    val_dataset = PreprocessedTokenDataset(val_files)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    print(f"Loaded {len(train_dataset.samples)} windows for training from {len(train_files)} songs")
    print(f"Loaded {len(val_dataset.samples)} windows for validation from {len(val_files)} songs")


    # Train Transformer
    print("Training Transformer model on dataset")
    train_losses, val_losses = train_on_dataset(
    model, train_loader, val_loader, optimizer, criterion, epochs=20, device=device)
    torch.save(model.state_dict(), "transformer.pth")
    wandb.save("transformer.pth")
    print("Model saved as transformer.pth")

    torch.save(train_losses, "train_losses.pt")
    torch.save(val_losses, "val_losses.pt")
    print("Losses saved in train_losses.pt and val_losses.pt")