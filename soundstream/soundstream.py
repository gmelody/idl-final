from functools import reduce
from typing import Literal
import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

from soundstream.encoder import Encoder
from soundstream.decoder import Decoder


class SoundStream(nn.Module):
    def __init__(self, n_q, codebook_size, D, C, strides=(2, 4, 5, 8)):
        super(SoundStream, self).__init__()

        # The temporal resampling ratio between input waveform and embeddings.
        # Not used in here, but helpful for consumers.
        self.M = reduce(lambda a, b: a * b, strides)

        self.encoder = Encoder(C=C, D=D, strides=strides)
        self.quantizer = ResidualVQ(
            num_quantizers=n_q,
            codebook_size=codebook_size,
            dim=D,
            kmeans_init=True,
            kmeans_iters=100,
            threshold_ema_dead_code=2,
        )
        self.decoder = Decoder(C=C, D=D, strides=strides)

    def decode_indices(self, indices):
        """
        indices: [B, T, Q] long tensors
        Reconstruct quantized latents the same way ResidualVQ does in forward().
        """

        quantizers = self.quantizer.layers if hasattr(self.quantizer, "layers") else self.quantizer.quantizers
        B, T, Q = indices.shape
        D = quantizers[0].codebook.weight.shape[1]

        latents = torch.zeros(B, T, D, device=indices.device)

        for q in range(Q):
            idx_q = indices[:, :, q]                      # [B, T]
            codebook = quantizers[q].codebook.weight      # [num_codes, D]
            embedded = codebook[idx_q]                    # [B, T, D]
            latents += embedded                           # accumulate residuals

        return latents

    def forward(
            self,
            x,
            mode: Literal['end-to-end', 'encode', 'decode'] = 'end-to-end',
        ):
        # x: batch_size x 1 x (T / 1)
        # e: batch_size x (T / M) x D --- where M is product of all numbers in `strides` tuple
        # o: batch_size x 1 x (T / 1)

        if mode == 'end-to-end':
            e = self.encoder(x)
            quantized, _, _ = self.quantizer(e.permute((0,2,1)))
            o = self.decoder(quantized.permute((0,2,1)))
            return o
        
        if mode == 'encode':
            e = self.encoder(x)
            # quantized, _, _ = self.quantizer(e.permute((0,2,1)))
            quantized, indicies, _ = self.quantizer(e.permute((0,2,1)))
            # print(quantized, indicies)
            return indicies
        
        if mode == 'decode':
            latents = self.decode_indices(x)             # [B, T, D]
            o = self.decoder(latents.permute(0, 2, 1))   # [B, 1, T]
            return o