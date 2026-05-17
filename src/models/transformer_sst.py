import torch
import torch.nn as nn


class TransformerSST2(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 64,
        num_heads: int = 8,
        num_layers: int = 2,
        max_len: int = 64,
        num_classes: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, hidden_dim, padding_idx=0)
        self.pos_emb = nn.Embedding(max_len, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=False,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = input_ids.shape

        pos = torch.arange(seq_len, device=input_ids.device)
        pos = pos.unsqueeze(0).expand(batch_size, seq_len)

        x = self.token_emb(input_ids) + self.pos_emb(pos)

        key_padding_mask = ~attention_mask.bool()

        x = self.encoder(x, src_key_padding_mask=key_padding_mask)

        mask = attention_mask.unsqueeze(-1).float()
        x = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)

        return self.classifier(x)