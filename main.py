import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from soundstream import from_pretrained
from transformer import Transformer
from torch.utils.data import random_split
import wandb



wandb.init(
    project="idl-final",         
    name="baseline-soundstream", 
    config={
        "learning_rate": 3e-4,
        "epochs": 5,
        "batch_size": 1,
        "optimizer": "Adam",
        "loss_fn": "CrossEntropy",
    }
)

# Dataloader
class AudioTokenDataset(Dataset):
    def __init__(self, root_dir, sound_stream):
        self.wav_files = glob.glob(os.path.join(root_dir, "**/*.wav"), recursive=True)
        self.wav_files += glob.glob(os.path.join(root_dir, "**/*.WAV"), recursive=True)
        self.sound_stream = sound_stream

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        path = self.wav_files[idx]
        # print(f"Encoding file {idx+1}/{len(self.wav_files)}: {os.path.basename(path)}", flush=True)
        waveform, sr = sf.read(path)

        # Convert stereo to mono if necessary
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)

        # Shape = (1, 1, n_samples)
        waveform = torch.tensor(waveform).unsqueeze(0).unsqueeze(0).float()

        quantized, indices = self.sound_stream(waveform, mode="encode")
        tokens = indices[:, :, 0].long().squeeze(0)
        x = tokens[:-1]
        y = tokens[1:]
        return x, y


# Training and Validation Loop
def train_on_dataset(model, train_loader, val_loader, optimizer, criterion, epochs=5, device="cpu"):
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
    data_dir = "maestro-v3.0.0"

    print("Loading SoundStream model...")
    sound_stream = from_pretrained()

    # Build dataset + loader
    dataset = AudioTokenDataset(data_dir, sound_stream)
    print(f"Found {len(dataset)} .wav files in {data_dir}")
    print("Example files:", dataset.wav_files[:5])

    # 80 20 training split
    train_size = int(0.80 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    print(f"Loaded {len(train_dataset)} audio files for training")
    print(f"Loaded {len(val_dataset)} audio files for validation")

    # Model setup
    model = Transformer()
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # Train Transformer
    print("Training Transformer model on dataset")
    train_losses, val_losses = train_on_dataset(
    model, train_loader, val_loader, optimizer, criterion, epochs=5, device=device)
    torch.save(model.state_dict(), "transformer.pth")
    wandb.save("transformer.pth")
    print("Model saved as transformer.pth")

    torch.save(train_losses, "train_losses.pt")
    torch.save(val_losses, "val_losses.pt")
    print("Losses saved in train_losses.pt and val_losses.pt")