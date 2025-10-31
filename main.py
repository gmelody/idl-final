import os
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf

from soundstream import from_pretrained
from transformer import Transformer, train, generate

# ==========================================
# 1. Dataset Class — handles all .wav files
# ==========================================
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


# ==========================================
# 2. Training Loop for Full Dataset
# ==========================================
def train_on_dataset(model, loader, optimizer, criterion, epochs=5, device="cpu"):
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch_idx, (x, y) in enumerate(loader):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            if (batch_idx + 1) % 5 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{epochs} | Average Loss: {avg_loss:.4f}")


# ==========================================
# 3. Main Script
# ==========================================
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
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    print(f"Loaded {len(dataset)} audio files for training")

    # Model setup
    vocab_size = 1024
    model = Transformer(vocab_size)
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    # Train Transformer
    print("Training Transformer model on dataset")
    train_on_dataset(model, loader, optimizer, criterion, epochs=5, device=device)
    torch.save(model.state_dict(), "transformer.pth")
    print("Model saved as transformer.pth")

    # # ==============================
    # # 4. Generation (test on one file)
    # # ==============================
    # print("\nGenerating new audio sample...")

    # # Load one example file to use as a prompt
    # test_file = glob.glob(os.path.join(data_dir, "**/*.wav"), recursive=True)[0]
    # waveform, sr = sf.read(test_file)
    # waveform = torch.tensor(waveform).unsqueeze(0).unsqueeze(0).float()
    # quantized, indices = sound_stream(waveform, mode="encode")
    # tokens = indices[:, :, 0].long()

    # start_seq = tokens[:, :100]
    # pred_tokens = generate(model, start_seq, max_new_tokens=200, temperature=0.8)

    # pred_expanded = pred_tokens.unsqueeze(-1).expand(-1, -1, quantized.shape[-1])
    # recovered = sound_stream(pred_expanded.float(), mode="decode")

    # import torchaudio
    # torchaudio.save("out.wav", recovered.squeeze().cpu(), 16000)
    # print("Generated audio saved as out.wav")