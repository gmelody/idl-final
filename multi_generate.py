import torch
import random
import soundfile as sf
from pathlib import Path
from soundstream import from_pretrained as load_soundstream
from multi_transformer import ContinuousTransformer  

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model(model_path):
    model = ContinuousTransformer(
        d_in=256,
        d_model=512,
        d_out=256,
        num_heads=8,
        num_layers=6,
        ff_dim=2048,
        dropout=0.1,
        max_seq_len=512
    )
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval().to(DEVICE)
    return model

@torch.no_grad()
def sample_sequence(model, seed_seq, steps):
    max_len = model.max_seq_len

    context = seed_seq.unsqueeze(0).to(DEVICE)   # [1, T, 256]
    full_seq = context.clone()                    # keeps everything

    for _ in range(steps):
        # keep context length <= max_seq_len
        if context.size(1) > max_len:
            context = context[:, -max_len:]

        out = model(context)                     # forward
        next_emb = out[:, -1:, :]                # [1, 1, 256]

        context = torch.cat([context, next_emb], dim=1)
        full_seq = torch.cat([full_seq, next_emb], dim=1)

    return full_seq.squeeze(0)

@torch.no_grad()
def decode_sequence(latents, sound_stream):
    """
    latents: [T, 256] continuous embeddings
    returns: waveform tensor [samples]
    """
    latents = latents.unsqueeze(0).to(DEVICE)  # [1, T, 256]
    audio = sound_stream(latents, mode="decode")  # [1, 1, samples]
    return audio.squeeze(0).squeeze(0).cpu()

if __name__ == "__main__":
    print("Loading model and SoundStream...")
    transformer = load_model("transformer.pth")

    sound_stream = load_soundstream().to(DEVICE)
    sound_stream.eval()

    # Load dataset
    dataset_folder = Path("2018_embedded")
    pt_files = list(dataset_folder.glob("*.pt"))
    selected_file = random.choice(pt_files)
    pt_data = torch.load(selected_file)
    seed_window = random.choice(pt_data['windows'])['x']

    print(f"Generating continuation from '{pt_data['name']}'")


    # Generate continuation
    full_sequence = sample_sequence(transformer, seed_window, steps=512)

    print("Frames:", full_sequence.shape[0])
    print("Seconds ≈", full_sequence.shape[0] / 128)
    print("Decoding audio from tokens")

    decoded_audio = decode_sequence(full_sequence, sound_stream)
    output_path = "generated_continuation.wav"
    # Convert tensor to NumPy array
    audio_np = decoded_audio.cpu().numpy()

    # If your waveform is 1D, reshape it to (N, 1) for sf
    if audio_np.ndim == 1:
        audio_np = audio_np.reshape(-1, 1)

    # Save using soundfile
    sf.write(output_path, audio_np, samplerate=24000)
    print(f"Saved generated continuation to '{output_path}'")    