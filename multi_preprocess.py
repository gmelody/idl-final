import os
import glob
import torch
import soundfile as sf
from soundstream import from_pretrained

def preprocess_wavs(
    input_dir,
    output_dir,
    max_length=2048,
    overlap=0.5,
    device="cuda",
    chunk_duration=15,
    num_quantizers_to_use=4,
):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading SoundStream model...")
    sound_stream = from_pretrained()
    sound_stream.to(device)
    sound_stream.eval()
    for param in sound_stream.parameters():
        param.requires_grad = False

    wav_files = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    wav_files += glob.glob(os.path.join(input_dir, "**/*.WAV"), recursive=True)
    print(f"Found {len(wav_files)} wav files")

    # Compute min window length
    strides = (2, 4, 5, 8)
    min_required_samples = 1
    for s in strides:
        min_required_samples *= s
    min_downsampled_frames = 100
    min_chunk_len = min_required_samples * min_downsampled_frames

    for i, wav_path in enumerate(wav_files):
        song_name = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nProcessing {i+1}/{len(wav_files)}: {song_name}")

        # Load waveform
        waveform, sr = sf.read(wav_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        waveform = torch.tensor(waveform).float().unsqueeze(0).unsqueeze(0).to(device)

        # Determine chunk size
        chunk_size = max(int(sr * chunk_duration), min_chunk_len)

        all_tokens = []  # will store [num_windows, T_down, X]
        all_full_tokens = []  # will store full 16 quantizers

        print("  Encoding...")

        for start in range(0, waveform.shape[-1], chunk_size):

            chunk = waveform[:, :, start:start+chunk_size]

            if chunk.shape[-1] < min_chunk_len:
                print("  Skipping short chunk")
                continue

            # Make divisible by stride
            valid_len = chunk.shape[-1] - (chunk.shape[-1] % min_required_samples)
            chunk = chunk[:, :, :valid_len]

            with torch.no_grad():
                indices_full = sound_stream(chunk, mode="encode")   # [1, T_down, 16]
            print("  Encoded chunk shape (full):", indices_full.shape)

            full_indices = indices_full.clone()

            # Slice top X quantizers
            indices = indices_full[:, :, :num_quantizers_to_use]    # [1, T_down, X]
            print("  Sliced chunk shape:", indices.shape)
            all_tokens.append(indices.cpu())
            all_full_tokens.append(full_indices.cpu())

        if len(all_tokens) == 0:
            print("  No tokens extracted")
            continue

        tokens = torch.cat(all_tokens, dim=1).squeeze(0)  # [T_total, X]
        full_tokens = torch.cat(all_full_tokens, dim=1).squeeze(0)  # [T_total, 16]
        print("  Tokens shape:", tokens.shape)
        total_frames = tokens.shape[0]

        if total_frames <= max_length:
            print("  Not enough tokens, skipping file.")
            continue

        # Build sliding windows
        step = int(max_length * (1 - overlap))

        windows = []
        for start in range(0, total_frames - max_length - 1, step):
            window = tokens[start:start+max_length+1]     # [L+1, X]
            window_full = full_tokens[start:start+max_length+1]  # [L+1, 16]
            # Multi-token autoregressive setup:
            # x = [L, X] input tokens for all quantizers
            # y = [L, X] next-timestep tokens for all quantizers
            x = window[:-1, :num_quantizers_to_use].clone()   # [L, X]
            print("  x window shape:", x.shape)
            y = window[1:,  :num_quantizers_to_use].clone()   # [L, X]
            print("  y window shape:", y.shape)
            full_16 = window_full[:-1, :].clone()  # [L, 16]
            windows.append({"x": x, "y": y, "full_16": full_16})

        out_file = os.path.join(output_dir, f"{song_name}.pt")
        torch.save({"name": song_name, "windows": windows}, out_file)

        print(f"  Saved {len(windows)} windows → {out_file}")

    print("Done.")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    preprocess_wavs(
        input_dir="2018",
        output_dir="2018_processed_all_levels",
        max_length=2048,
        overlap=0.5,
        device="cuda",
        chunk_duration=15,
        num_quantizers_to_use=4,
    )