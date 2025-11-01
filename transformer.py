import torch
import torch.nn as nn
import torch.nn.functional as F

class Transformer(nn.Module):
    """
    Transformer model for sequence modeling.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of the embeddings.
        num_heads (int): Number of attention heads.
        num_layers (int): Number of Transformer encoder layers.
        ff_dim (int): Dimension of the feedforward network.
        dropout (float): Dropout probability.
        max_seq_len (int): Maximum sequence length.
    """
    def __init__(self, vocab_size=1024, embed_dim=512, num_heads=8, num_layers=6, ff_dim=2048, dropout=0.1, max_seq_len=20000):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
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
        self.head = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):        
        batch_size, seq_len = x.shape
        pos_ids = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = self.token_emb(x) + self.pos_emb(pos_ids)
        x = self.dropout(x)

        # causal mask (prevent attention to future positions)
        mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        x = self.transformer(x, mask=mask)

        x = self.ln(x)
        logits = self.head(x)
        return logits

def generate(model, start_seq, max_new_tokens=50, temperature=1.0):
    model.eval()
    seq = start_seq.clone()

    for _ in range(max_new_tokens):
        logits = model(seq)
        logits = logits[:, -1, :] / temperature
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        seq = torch.cat([seq, next_token], dim=1)

    return seq
