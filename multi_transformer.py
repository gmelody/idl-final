import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiQuantizerTransformer(nn.Module):
    """
    Transformer predicting per-quantizer discrete codes.
    
    Output shape: [batch, seq_len, num_quantizers, codebook_size]
    """

    def __init__(
        self,
        num_quantizers,
        codebook_size,
        embed_dim=512,
        num_heads=8,
        num_layers=6,
        ff_dim=2048,
        dropout=0.1,
        max_seq_len=2048
    ):
        super().__init__()

        self.num_quantizers = num_quantizers
        self.codebook_size = codebook_size
        self.vocab_size = codebook_size      # per quantizer
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        # We embed each quantizer token separately:
        # input tokens will be shaped [B, T, Q]
        self.token_emb = nn.Embedding(codebook_size, embed_dim)
        self.pos_emb = nn.Embedding(max_seq_len, embed_dim)

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln = nn.LayerNorm(embed_dim)

        # Final head predicts num_quantizers × codebook_size logits
        self.head = nn.Linear(embed_dim, num_quantizers * codebook_size)

    def forward(self, x):
        """
        x shape: [batch, seq_len, num_quantizers]
        """

        B, T, Q = x.shape
        assert Q == self.num_quantizers

        # Embed each quantizer token and sum them (common trick)
        # token_emb(x) -> [B, T, Q, D]
        x = self.token_emb(x)
        x = x.sum(dim=2)  # [B, T, D]

        # Add positional encoding
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_emb(pos)
        x = self.dropout(x)

        # causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        x = self.transformer(x, mask=mask)
        x = self.ln(x)

        # Predict logits
        logits = self.head(x)  # [B, T, Q*V]
        logits = logits.view(B, T, self.num_quantizers, self.codebook_size)

        return logits

@torch.no_grad()
def generate(model, start_tokens, max_new_tokens=50, temperature=1.0):
    """
    start_tokens: [B, T0, Q]
    returns: [B, T0 + max_new_tokens, Q]
    """

    model.eval()
    seq = start_tokens.clone()  # [B, T0, Q]
    B, _, Q = seq.shape

    for _ in range(max_new_tokens):
        logits = model(seq)              # [B, T, Q, V]
        logits_last = logits[:, -1]      # [B, Q, V]

        # temperature scaling
        logits_last = logits_last / temperature

        # softmax per quantizer
        probs = F.softmax(logits_last, dim=-1).view(B * Q, model.codebook_size)

        # sample per quantizer independently
        next_tokens = torch.multinomial(probs, 1).view(B, Q)  # [B, Q]

        # append to sequence
        seq = torch.cat([seq, next_tokens.unsqueeze(1)], dim=1)

    return seq
