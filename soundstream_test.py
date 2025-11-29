import torchaudio

from soundstream import from_pretrained, load
from vector_quantize_pytorch import ResidualVQ

waveform = load('piano_melody.wav')
audio_codec = from_pretrained()  # downloads model from Hugging Face

indices = audio_codec(waveform, mode='encode')
print(indices.shape)
# quantized_latents = ResidualVQ.get_codes_from_indices(indices)

recovered = audio_codec(indices, mode='decode')

# torchaudio.save('out2.wav', recovered[0], 16000)