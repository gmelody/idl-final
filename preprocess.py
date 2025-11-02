import os
import glob
import argparse
import torch
import soundfile as sf
from soundstream import from_pretrained


def preprocess_wavs(
    input_dir,
    output_dir,
    max_length=2048,
    overlap=0.5,
    device="cpu"
):
    os.makedirs(output_dir, exist_ok=True)

    # Load SoundStream model (frozen)
    print("Loading SoundStream model...")
    sound_stream = from_pretrained()
    sound_stream.to(device)
    sound_stream.eval()
    for param in sound_stream.parameters():
        param.requires_grad = False

    # Find WAV files
    wav_files = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    wav_files += glob.glob(os.path.join(input_dir, "**/*.WAV"), recursive=True)
    print(f"Found {len(wav_files)} wav files in {input_dir}")

    file_counter = 0
    total_window_counter = 0

    for wav_path in wav_files:
        file_counter += 1
        song_name = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nProcessing file {file_counter}/{len(wav_files)}: {song_name}")

        # Load WAV and convert to mono if necessary
        waveform, sr = sf.read(wav_path)
        if len(waveform.shape) > 1:
            waveform = waveform.mean(axis=1)
        waveform = torch.tensor(waveform).unsqueeze(0).unsqueeze(0).float().to(device)

        # Encode tokens
        with torch.no_grad():
            indices = sound_stream(waveform, mode="encode")  # (1, T, Q)
        tokens = indices[:, :, 0].long().squeeze(0)  # (T,)
        total_tokens = len(tokens)

        # Sliding window setup
        step = int(max_length * (1 - overlap))
        song_windows = []
        num_windows = 0

        for start in range(0, total_tokens - max_length - 1, step):
            window = tokens[start:start + max_length + 1]
            x = window[:-1]
            y = window[1:]
            song_windows.append({"x": x, "y": y})
            num_windows += 1

        # Save all windows for this song in a single .pt file
        save_path = os.path.join(output_dir, f"{song_name}.pt")
        torch.save({
            "name": song_name,
            "windows": song_windows
        }, save_path)

        total_window_counter += num_windows
        print(f"  {num_windows} windows saved for this file -> {save_path}")

    print(f"Completed: {total_window_counter} total windows saved in {output_dir}")

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    preprocess_wavs(
        input_dir= "maestro-v3.0.0/2018",
        output_dir= "2018_processed",
        max_length= 2048,
        overlap= 0.5,
        device= device
    )
