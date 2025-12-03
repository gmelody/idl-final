import torch
import torch.nn as nn

class ContinuousTransformer(nn.Module):
    def __init__(
        self,
        d_in=256,
        d_model=512,
        d_out=256,
        num_heads=8,
        num_layers=6,
        ff_dim=2048,
        dropout=0.1,
        max_seq_len=512,
    ):
        super().__init__()

        self.d_in = d_in
        self.d_model = d_model
        self.d_out = d_out
        self.max_seq_len = max_seq_len

        # Project input embeddings into transformer space
        self.input_proj = nn.Linear(d_in, d_model)

        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln = nn.LayerNorm(d_model)

        # Project back to embedding space
        self.output_proj = nn.Linear(d_model, d_out)

    def forward(self, x):
        """
        x: [B, T, D_in]
        returns: [B, T, D_out]
        """
        B, T, D = x.shape
        assert D == self.d_in, f"Expected D={self.d_in}, got {D}"
        assert T <= self.max_seq_len, f"T={T} > max_seq_len={self.max_seq_len}"

        # Input projection
        h = self.input_proj(x)    # [B, T, d_model]

        # Positional embeddings
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = h + self.pos_emb(pos)
        h = self.dropout(h)

        # Causal mask
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Transformer
        h = self.transformer(h, mask=mask)
        h = self.ln(h)

        # Predict next-step embeddings
        out = self.output_proj(h)   # [B, T, d_out]
        return out