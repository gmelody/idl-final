import torch

pt = torch.load("2018_processed_four_levels/MIDI-Unprocessed_Schubert7-9_MID--AUDIO_16_R2_2018_wav.pt")
first_window = pt["windows"][0]["x"]
print("Window shape:", first_window.shape)