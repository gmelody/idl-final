import torch
import random
import soundfile as sf
from pathlib import Path
from soundstream import from_pretrained as load_soundstream
from multi_transformer import MultiQuantizerTransformer    # ← your 4-token transformer
import numpy as np

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Load trained transformer
# -----------------------------
def load_model(model_path, vocab_size=1024, num_q=4, seq_len=2048):
    model = MultiQuantizerTransformer(
        num_quantizers = num_q,
        codebook_size = vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        ff_dim=2048,
        dropout=0.1,
        max_seq_len=2048
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


# -----------------------------
# Sampling for multi-token prediction
# -----------------------------
def sample_sequence(model, seed_seq, max_new_tokens=500, temperature=1.0):
    model.eval()

    num_q = seed_seq.shape[1]      # = 4
    vocab_size = model.vocab_size
    max_len = model.max_seq_len    # 2048

    # Ensure seed indices are in valid integer range
    context = torch.clamp(seed_seq.clone().to(DEVICE), 0, vocab_size - 1)

    for _ in range(max_new_tokens):

        # Keep the last max_len frames (sliding window)
        if context.shape[0] > max_len:
            context = context[-max_len:]

        # Model expects [B, T, 4]
        logits = model(context.unsqueeze(0))  # → [1, T, 4, vocab]
        logits_last = logits[:, -1] / temperature  # → [1, 4, vocab]

        next_tokens = []
        for q in range(num_q):
            probs = torch.softmax(logits_last[0, q], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
            next_tokens.append(next_token)

        next_tokens = torch.tensor(next_tokens, device=DEVICE).unsqueeze(0)  # [1, 4]

        # Append new [4] token row
        context = torch.cat([context, next_tokens], dim=0)

    return context  # [T + new, 4]


# -----------------------------
# Decoder: tokens → embeddings → waveform
# -----------------------------
def decode_tokens(tokens_4q, sound_stream, seed_full):
    """
    tokens_4q: [T_new, 4] — predicted tokens
    seed_full: [T_seed, 16] — from original SoundStream encoding (full 16 quantizers)
    """

    tokens_4q = tokens_4q.to(DEVICE)
    T_new = tokens_4q.shape[0]

    # Build full 16-quantizer matrix
    full_idx = torch.zeros(1, T_new, 16, dtype=torch.long, device=DEVICE)
    full_idx[:, :, :4] = tokens_4q

    # --- copy the true quantizers q4–q15 from seed_full ---
    seed_q = seed_full[:, 4:]                  # [T_seed, 12]
    seed_q_expanded = seed_q[:T_new]                # crop to match length
    full_idx[:, :, 4:] = seed_q_expanded.unsqueeze(0)

    # --- manual decoding ---
    D = sound_stream.quantizer.layers[0].codebook.shape[1]
    num_q = sound_stream.quantizer.num_quantizers

    residual = torch.zeros(1, T_new, D, device=DEVICE)

    for q in range(num_q):
        idx = full_idx[:, :, q]
        codebook = sound_stream.quantizer.layers[q].codebook
        latent = codebook[idx]
        residual += latent

    decoded = sound_stream.decoder(residual.permute(0, 2, 1))
    return decoded.squeeze(0).cpu()


# -----------------------------
# MAIN SCRIPT
# -----------------------------
if __name__ == "__main__":
    print("Loading model and SoundStream...")

    transformer = load_model("transformer.pth")   # <-- your new 4q trained model
    sound_stream = load_soundstream().to(DEVICE)
    sound_stream.eval()

    # Load a random seed window from your 4-level dataset
    dataset_folder = Path("2018_processed_four_levels")
    pt_files = list(dataset_folder.glob("*.pt"))
    selected_pt = random.choice(pt_files)
    pt_data = torch.load(selected_pt)

    seed_entry = random.choice(pt_data["windows"])
    seed_window = seed_entry["x"]        # [L, 4]
    seed_full   = seed_entry["full_16"]  # [L, 16]

    print(f"Generating continuation starting from {pt_data['name']}")

    # Generate new continuation
    full_tokens = sample_sequence(
        transformer,
        seed_seq=seed_window,
        max_new_tokens=500,
        temperature=1.1
    )

    print("Decoding audio...")
    audio = decode_tokens(full_tokens, sound_stream, seed_full)

    # Save WAV
    # ---- FIX WAV OUTPUT SHAPE ----
    audio_np = audio.squeeze().cpu().detach().numpy()

    # ------------------------------
    # APPLY LOW-PASS FILTER HERE
    # ------------------------------
    import torchaudio
    import torch

    audio_tensor = torch.from_numpy(audio_np).unsqueeze(0)  # shape: [1, T]
    audio_tensor = torchaudio.functional.lowpass_biquad(
        audio_tensor,
        sample_rate=24000,
        cutoff_freq=7000,   # <-- you can adjust this
    )

    # convert back to numpy
    audio_np = audio_tensor.squeeze().numpy()
    # ------------------------------

    sf.write("generated_4quantizer.wav", audio_np, samplerate=24000)