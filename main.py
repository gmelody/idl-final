import torchaudio

from soundstream import from_pretrained, load


waveform = load('calm_piano.wav')
audio_codec = from_pretrained()  # downloads model from Hugging Face

quantized = audio_codec(waveform, mode='encode')
recovered = audio_codec(quantized, mode='decode')

torchaudio.save('out.wav', recovered[0], 16000)