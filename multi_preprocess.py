import os
import glob
import torch
import soundfile as sf

from soundstream import from_pretrained   # your helper

def preprocess_wavs(
    input_dir,
    output_dir,
    max_length=512,          # sequence length in *embedding* timesteps
    overlap=0.5,
    device="cuda",
    chunk_duration=15.0,     # seconds of raw audio per chunk
):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading SoundStream model...")
    sound_stream = from_pretrained()
    sound_stream.to(device)
    sound_stream.eval()
    for p in sound_stream.parameters():
        p.requires_grad = False

    wav_files = glob.glob(os.path.join(input_dir, "**/*.wav"), recursive=True)
    wav_files += glob.glob(os.path.join(input_dir, "**/*.WAV"), recursive=True)
    print(f"Found {len(wav_files)} wav files in {input_dir}")

    # same stride logic as before (only for min chunk size)
    strides = (2, 4, 5, 8)
    min_required_samples = 1
    for s in strides:
        min_required_samples *= s      # 2*4*5*8 = 320

    min_downsampled_frames = 100
    min_chunk_len = min_required_samples * min_downsampled_frames

    for i, wav_path in enumerate(wav_files):
        song_name = os.path.splitext(os.path.basename(wav_path))[0]
        print(f"\nProcessing file {i+1}/{len(wav_files)}: {song_name}")

        waveform, sr = sf.read(wav_path)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)  # stereo → mono

        waveform = torch.tensor(waveform, dtype=torch.float32, device=device)
        waveform = waveform.unsqueeze(0).unsqueeze(0)   # [1, 1, T]

        chunk_size = max(int(sr * chunk_duration), min_chunk_len)

        all_embs = []   # list of [T_chunk_down, D]

        print("  Encoding to quantized embeddings...")
        for start in range(0, waveform.shape[-1], chunk_size):
            chunk = waveform[:, :, start:start+chunk_size]
            if chunk.shape[-1] < min_chunk_len:
                print("  Skipping short chunk.")
                continue

            valid_len = chunk.shape[-1] - (chunk.shape[-1] % min_required_samples)
            chunk = chunk[:, :, :valid_len]

            with torch.no_grad():
                # quantized: [1, T_down, D]
                quantized = sound_stream(chunk, mode="encode")

            all_embs.append(quantized.squeeze(0).cpu())  # [T_down, D]

        if not all_embs:
            print("  No embeddings extracted, skipping file.")
            continue

        embs = torch.cat(all_embs, dim=0)   # [T_total, D]
        T_total, D = embs.shape
        print(f"  Total embedding frames: {T_total}, dim: {D}")

        if T_total <= max_length:
            print("  Not enough frames for one window, skipping file.")
            continue

        # Sliding windows in embedding space
        step = int(max_length * (1 - overlap))
        windows = []

        for start in range(0, T_total - max_length - 1, step):
            window = embs[start:start + max_length + 1]    # [L+1, D]
            x = window[:-1]    # [L, D] input
            y = window[1:]     # [L, D] target (next-step embedding)
            windows.append({"x": x, "y": y})

        save_path = os.path.join(output_dir, f"{song_name}.pt")
        torch.save({"name": song_name, "windows": windows}, save_path)
        print(f"  Saved {len(windows)} windows → {save_path}")

    print("Done preprocessing.")
    

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    preprocess_wavs(
        input_dir="2018",
        output_dir="2018_embedded",
        max_length=512,     # smaller than 2048 → easier training
        overlap=0.5,
        device=device,
        chunk_duration=15.0,
    )