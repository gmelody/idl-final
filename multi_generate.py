import torch
import random
import soundfile as sf
from pathlib import Path
from soundstream import from_pretrained as load_soundstream
from transformer import Transformer  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path, vocab_size=1024, seq_len=2048):
    model = Transformer(
        vocab_size=vocab_size,
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

def sample_sequence(model, seed_seq, max_new_tokens=500, temperature=1.0):
    model.eval()
    vocab_size = model.vocab_size
    max_seq_len = model.max_seq_len  # Add this to your Transformer if not present

    # Clamp the seed sequence to ensure all indices are valid
    context = torch.clamp(seed_seq.clone().to(DEVICE), min=0, max=vocab_size - 1)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            # Roll context if too long (keep last window of max_seq_len)
            if context.size(0) > max_seq_len:
                context = context[-max_seq_len:]

            logits = model(context.unsqueeze(0))[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Clamp next token to valid range
            next_token = torch.clamp(next_token, min=0, max=vocab_size - 1)

            context = torch.cat([context, next_token.squeeze(0)], dim=0)

    return context

def decode_tokens(tokens, sound_stream):
    with torch.no_grad():
        print("TOKEN SHAPE: ", tokens.shape)
        tokens = tokens.to(DEVICE).unsqueeze(0)  # [1, T, Q]
        # quantizer = sound_stream.quantizer

        # Decode embeddings
        decoded_audio = sound_stream.decoder(tokens.permute(0, 2, 1))  # [1, D, T]

    return decoded_audio.squeeze().cpu()

if __name__ == "__main__":
    print("Loading model and SoundStream...")
    transformer = load_model("transformer.pth")

    sound_stream = load_soundstream().to(DEVICE)
    sound_stream.eval()

    # Load dataset
    dataset_folder = Path("2018_processed")
    pt_files = list(dataset_folder.glob("*.pt"))
    selected_file = random.choice(pt_files)
    pt_data = torch.load(selected_file)
    seed_window = random.choice(pt_data['windows'])['x']

    print(f"Generating continuation from '{pt_data['name']}'")


    # Generate continuation
    full_sequence = sample_sequence(
        model=transformer,
        seed_seq=seed_window,
        max_new_tokens=500,
        temperature=1.2
    )

    print("Decoding audio from tokens")
    decoded_audio = decode_tokens(full_sequence, sound_stream)
    output_path = "generated_continuation.wav"
    # Convert tensor to NumPy array
    audio_np = decoded_audio.cpu().numpy()

    # If your waveform is 1D, reshape it to (N, 1) for sf
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(-1, 1)

    # Save using soundfile
    sf.write(output_path, audio_np, samplerate=24000)
    print(f"Saved generated continuation to '{output_path}'")    