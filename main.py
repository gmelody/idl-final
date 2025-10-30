import torchaudio
import torch.nn as nn
import torch.optim as optim

from soundstream import from_pretrained, load
from transformer import *

waveform = load('calm_piano.wav')
curr_model = None
audio_codec = from_pretrained()  # downloads model from Hugging Face

# encode from wav to token space
indices = audio_codec(waveform, mode='encode')
print(indices.min(), indices.max())
print(indices.shape)
tokens = indices.reshape(1, 640*16)  

# continue the song with transformer
vocab_size = 1024
# either train new model or import existing

x = tokens[:-1, :].long()
y = tokens[1:, :].long()

if not curr_model:
  model = Transformer(vocab_size)
  optimizer = optim.Adam(model.parameters(), lr=3e-4)
  criterion = nn.CrossEntropyLoss()

  train(x, y, model, optimizer, criterion, epochs=5)
  torch.save(model.state_dict(), 'transformer.pth')
else:
  # import model
  model = torch.load('transformer.pth')

# pred = generate(model, indices)


# # decode from token space to wav
# recovered = audio_codec(pred, mode='decode')

# torchaudio.save('out.wav', recovered[0], 16000)
