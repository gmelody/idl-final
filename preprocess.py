import os
import glob
import torch
import soundfile as sf
from soundstream import from_pretrained

def preprocess_wavs(input_dir, output_dir, max_length=2048, overlap=0.5, device="cuda", chunk_duration=15):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading SoundStream model...")
    sound_stream = from_pretrained()
    sound_stream.to(device)
    sound_stream.eval()
    for param in sound_stream.parameters():
        param.requires_grad = False

    wav_files = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    wav_files += glob.glob(os.path.join(input_dir, "**/*.WAV"), recursive=True)
    print(f"Found {len(wav_files)} wav files in {input_dir}")

    # Encoding constraints
    strides = (2, 4, 5, 8)
    min_required_samples = 1
    for s in strides:
        min_required_samples *= s    # 2 * 4 * 5 * 8 = 320

    # Ensure we have enough downsampled steps to survive SoundStream encoder
    min_downsampled_frames = 100
    min_chunk_len = min_required_samples * min_downsampled_frames

    for i, wav_path in enumerate(wav_files):
        song_name = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nProcessing file {i+1}/{len(wav_files)}: {song_name}")

        waveform, sr = sf.read(wav_path)
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)  # Stereo to mono
        waveform = torch.tensor(waveform).unsqueeze(0).unsqueeze(0).float().to(device)

        # Enforce safe chunk size
        chunk_size = max(int(sr * chunk_duration), min_chunk_len)
        tokens = []

        print("  Encoding audio in chunks...")

        for start in range(0, waveform.shape[-1], chunk_size):
            chunk = waveform[:, :, start:start + chunk_size]

            if chunk.shape[-1] < min_chunk_len:
                print("  Skipping chunk (too short for encoder).")
                continue

            # Trim chunk so it's divisible by total stride
            valid_chunk_length = chunk.shape[-1] - (chunk.shape[-1] % min_required_samples)
            chunk = chunk[:, :, :valid_chunk_length].to(device)

            with torch.no_grad():
                indices = sound_stream(chunk, mode="encode")

            chunk_tokens = indices[:, :, 0].long().squeeze(0).cpu()
            tokens.append(chunk_tokens)

        tokens = torch.cat(tokens) if tokens else torch.tensor([], dtype=torch.long)
        total_tokens = len(tokens)

        # Skip songs that didn't produce any tokens
        if total_tokens <= max_length:
            print("  Not enough tokens to create even 1 window, skipping file.")
            continue

        # Windowing over tokens
        step = int(max_length * (1 - overlap))
        song_windows = []
        for start in range(0, total_tokens - max_length - 1, step):
            window = tokens[start:start + max_length + 1]
            x = window[:-1]
            y = window[1:]
            song_windows.append({"x": x, "y": y})

        save_path = os.path.join(output_dir, f"{song_name}.pt")
        torch.save({"name": song_name, "windows": song_windows}, save_path)
        print(f"  Saved {len(song_windows)} windows -> {save_path}")

    print("Done preprocessing")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    preprocess_wavs(
        input_dir="2018",
        output_dir="2018_processed",
        max_length=2048,
        overlap=0.5,
        device=device
    )