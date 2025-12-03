import torch
import random
import soundfile as sf
from pathlib import Path
from soundstream import from_pretrained as load_soundstream
from multi_transformer import MultiQuantizerTransformer
import julius.filters

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------
# Load Transformer Model
# ---------------------------------------------------------
def load_model(model_path, vocab_size=1024, seq_len=2048):
    model = MultiQuantizerTransformer(
        num_quantizers=4,
        codebook_size=vocab_size,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        ff_dim=2048,
        dropout=0.1,
        max_seq_len=seq_len
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model


# ---------------------------------------------------------
# Autoregressive Sampling
# ---------------------------------------------------------
def sample_sequence(model, seed_seq, max_new_tokens=500, temperature=1.0):
    model.eval()
    num_q = seed_seq.shape[1]  # 4
    vocab_size = model.vocab_size
    max_len = model.max_seq_len

    context = torch.clamp(seed_seq.clone().to(DEVICE), 0, vocab_size - 1)
    full_seq = context.clone()

    for _ in range(max_new_tokens):
        if context.size(0) > max_len:
            context = context[-max_len:]

        logits = model(context.unsqueeze(0))  # [1, T, 4, vocab]
        logits_last = logits[:, -1] / temperature  # [1, 4, vocab]

        next_tokens = []
        for q in range(num_q):
            probs = torch.softmax(logits_last[0, q], dim=-1)
            next_tokens.append(torch.multinomial(probs, 1).item())

        next_tokens = torch.tensor(next_tokens, device=DEVICE).unsqueeze(0)
        context = torch.cat([context, next_tokens], dim=0)
        full_seq = torch.cat([full_seq, next_tokens], dim=0)

    return full_seq


# ---------------------------------------------------------
# Convert 16-level indices → latent embedding → decoder audio
# ---------------------------------------------------------
def indices_to_latent(full_idx_16, quantizer):
    """
    full_idx_16: [1, T, 16] long
    quantizer: SoundStream.ResidualVQ()
    Returns latent: [1, D, T]
    """
    B, T, Q = full_idx_16.shape
    codebooks = quantizer.codebooks  # list of 16 nn.Embedding
    D = codebooks[0].weight.size(1)

    latent = torch.zeros((B, D, T), device=full_idx_16.device)

    for q in range(Q):
        idx_q = full_idx_16[:, :, q]         # [B, T]
        emb_q = codebooks[q](idx_q)          # [B, T, D]
        latent += emb_q.permute(0, 2, 1)     # sum residual codebooks

    return latent


# ---------------------------------------------------------
# Option B — Decode continuation
# ---------------------------------------------------------
def decode_with_option_B(pred_q0_3, seed_full16, sound_stream, chunk_frames=300, overlap=50):
    """
    Chunked decoding to prevent CUDA OOM.
    pred_q0_3: [T_total, 4]
    seed_full16: [1, T_seed, 16]
    """

    with torch.no_grad():
        pred_q0_3 = pred_q0_3.to(DEVICE)
        seed_full16 = seed_full16.to(DEVICE)

        T_seed = seed_full16.size(1)
        T_total = pred_q0_3.size(0)
        T_gen = T_total - T_seed

        # --- Build [1, T_total, 16] full index tensor ---
        full_idx = torch.zeros((1, T_total, 16), dtype=torch.long, device=DEVICE)

        # Fill Q0–3 predictions
        full_idx[0, :, 0:4] = pred_q0_3

        # Original Q4–15 for seed
        full_idx[0, :T_seed, 4:16] = seed_full16[0, :, 4:16]

        # Repeat last Q4–15 for continuation
        last_q4_15 = seed_full16[0, -1, 4:16]
        full_idx[0, T_seed:, 4:16] = last_q4_15.unsqueeze(0).expand(T_gen, -1)

        # ------------------------------------------------------
        # Convert indices → latent
        # latent: [1, D, T_total]
        # ------------------------------------------------------
        latent = indices_to_latent(full_idx, sound_stream.quantizer)

        # ------------------------------------------------------
        # Chunked decode
        # ------------------------------------------------------
        decoded_chunks = []
        T = latent.shape[-1]

        start = 0
        while start < T:
            end = min(start + chunk_frames, T)

            # Select latent slice
            latent_chunk = latent[:, :, start:end]  # [1, D, chunk]

            # Decode small chunk
            audio_chunk = sound_stream.decoder(latent_chunk)  # [1, 1, samples]
            audio_chunk = audio_chunk.squeeze(0).squeeze(0)   # [samples]

            decoded_chunks.append(audio_chunk)

            start = end - overlap  # overlap for continuity

        # Concatenate decoded audio
        decoded_audio = torch.cat(decoded_chunks, dim=0)

        return decoded_audio.cpu()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    print("Loading model and SoundStream...")
    transformer = load_model("four_quant_model.pth")
    sound_stream = load_soundstream().to(DEVICE)
    sound_stream.eval()

    # Select random token window
    dataset_folder = Path("2018_processed_four_levels")
    pt_files = list(dataset_folder.glob("*.pt"))
    selected_file = random.choice(pt_files)
    pt_data = torch.load(selected_file)
    seed_window = random.choice(pt_data['windows'])['x']  # [T_seed, 4]

    print(f"Generating continuation from '{pt_data['name']}'")

    # Load original WAV and re-encode to full 16 quantizers
    wav_path_guess = list(Path("2018").glob(f"{pt_data['name']}.*"))[0]
    wav, sr = sf.read(wav_path_guess)
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    wav_t = torch.tensor(wav).float().unsqueeze(0).unsqueeze(0).to(DEVICE)

    seed_full16 = sound_stream(wav_t, mode="encode")[:, :seed_window.shape[0]]  # [1, T_seed, 16]

    # Generate continuation
    full_sequence = sample_sequence(
        transformer, seed_window, max_new_tokens=500, temperature=1.2
    )
    print("FULL SEQUENCE SHAPE:", full_sequence.shape)

    # Decode with Option B
    print("Decoding continuation...")
    decoded_audio = decode_with_option_B(full_sequence, seed_full16, sound_stream)

    # Light smoothing
    lowpass = julius.filters.BandPassFilter(cutoff_low=0, cutoff_high=0.05, stride=1, pad=True)
    decoded_audio = lowpass(decoded_audio)

    audio_np = decoded_audio.numpy()
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(-1, 1)

    sf.write("generated_continuation_new.wav", audio_np, samplerate=24000)
    print("Saved generated_continuation_new.wav")