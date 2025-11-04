import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

audio, sr = librosa.load("generated_continuation.wav", sr=None)
mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_fft=1024, hop_length=256, n_mels=128)
mel_db = librosa.power_to_db(mel, ref=np.max)

plt.figure(figsize=(10, 4))
librosa.display.specshow(mel_db, sr=sr, hop_length=256, x_axis='time', y_axis='mel', cmap='magma')
plt.colorbar(format="%+2.0f dB")
plt.title("Mel Spectrogram")
plt.tight_layout()
plt.show()