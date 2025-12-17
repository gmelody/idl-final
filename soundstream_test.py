import torch
import torchaudio

from soundstream import from_pretrained

# ---- LOAD SOUNDSTREAM ----
audio_codec = from_pretrained().to("cuda")
audio_codec.eval()

# ---- LOAD ONE WINDOW ----
data = torch.load("2018_embedded/MIDI-Unprocessed_Schubert7-9_MID--AUDIO_16_R2_2018_wav.pt")
window = data["windows"][100]

# quantized is embeddings: [T, D]
quantized = window["x"]      # shape [512, 256]
print("loaded:", quantized.shape)

# ---- RESHAPE FOR DECODE ----
# SoundStream decode requires [B, T, D]
quantized = quantized.unsqueeze(0).to("cuda").float()
print("reshaped for decode:", quantized.shape)   # [1, 512, 256]

# ---- DECODE ----
audio = audio_codec(quantized, mode="decode")   # → [1, 1, samples]
audio = audio.squeeze().cpu().detach()


# ---- SAVE ----
torchaudio.save("decoded_test.wav", audio.unsqueeze(0), 24000)
print("Saved decoded_test.wav")