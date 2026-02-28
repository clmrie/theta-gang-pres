import math
import torch
import torch.nn as nn

from geometry import N_ZONES


class SpikeEncoder(nn.Module):
    """Encodes spike waveforms (channels, 32 samples) -> embed_dim via CNN."""

    def __init__(self, n_channels, embed_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_channels, 32, kernel_size=5, padding=2), nn.ReLU(),
            nn.Conv1d(32, embed_dim, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x):
        return self.conv(x).squeeze(-1)


class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for Transformer."""

    def __init__(self, embed_dim, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class SpikeTransformerHierarchical(nn.Module):
    """Hierarchical Transformer: 3-zone classification + conditional regression.

    Includes spike dropout and Gaussian noise as data augmentation during training.
    """

    def __init__(self, nGroups, nChannelsPerGroup, n_zones=N_ZONES,
                 embed_dim=64, nhead=4, num_layers=2, dropout=0.2,
                 spike_dropout=0.15, noise_std=0.5, max_channels=6):
        super().__init__()
        self.nGroups = nGroups
        self.embed_dim = embed_dim
        self.n_zones = n_zones
        self.max_channels = max_channels
        self.spike_dropout = spike_dropout
        self.noise_std = noise_std

        # One encoder per electrode group
        self.spike_encoders = nn.ModuleList([
            SpikeEncoder(max_channels, embed_dim) for _ in range(nGroups)
        ])
        self.shank_embedding = nn.Embedding(nGroups, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 4,
            dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False,
        )

        # Zone classification head
        self.cls_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, n_zones),
        )

        # Per-zone regression heads (mu and log_sigma)
        self.mu_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(embed_dim, 2),
            ) for _ in range(n_zones)
        ])
        self.log_sigma_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(embed_dim, 2),
            ) for _ in range(n_zones)
        ])

        # Curvilinear distance head
        self.d_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(embed_dim, 1), nn.Sigmoid(),
        )

    def _apply_spike_dropout(self, mask):
        if not self.training or self.spike_dropout <= 0:
            return mask
        drop_mask = torch.rand_like(mask.float()) < self.spike_dropout
        active = ~mask
        new_drops = drop_mask & active
        remaining = active & ~new_drops
        all_dropped = remaining.sum(dim=1) == 0
        if all_dropped.any():
            new_drops[all_dropped] = False
        return mask | new_drops

    def _apply_waveform_noise(self, waveforms):
        if not self.training or self.noise_std <= 0:
            return waveforms
        return waveforms + torch.randn_like(waveforms) * self.noise_std

    def _encode(self, waveforms, shank_ids, mask):
        """Shared backbone: encode -> transformer -> pooling."""
        batch_size, seq_len = waveforms.shape[:2]
        mask = self._apply_spike_dropout(mask)
        waveforms = self._apply_waveform_noise(waveforms)

        embeddings = torch.zeros(batch_size, seq_len, self.embed_dim, device=waveforms.device)
        for g in range(self.nGroups):
            group_mask = (shank_ids == g) & (~mask)
            if group_mask.any():
                embeddings[group_mask] = self.spike_encoders[g](waveforms[group_mask])

        embeddings = embeddings + self.shank_embedding(shank_ids)
        embeddings = self.pos_encoding(embeddings)
        encoded = self.transformer(embeddings, src_key_padding_mask=mask)

        active_mask = (~mask).unsqueeze(-1).float()
        pooled = (encoded * active_mask).sum(dim=1) / (active_mask.sum(dim=1) + 1e-8)
        return pooled

    def forward(self, waveforms, shank_ids, mask):
        pooled = self._encode(waveforms, shank_ids, mask)
        cls_logits = self.cls_head(pooled)
        mus = [head(pooled) for head in self.mu_heads]
        sigmas = [torch.exp(head(pooled)) for head in self.log_sigma_heads]
        d_pred = self.d_head(pooled)
        return cls_logits, mus, sigmas, d_pred

    def predict(self, waveforms, shank_ids, mask):
        """Combined prediction via softmax-weighted mixture of zone heads."""
        cls_logits, mus, sigmas, d_pred = self.forward(waveforms, shank_ids, mask)
        probs = torch.softmax(cls_logits, dim=1)  # (batch, 3)

        mu_stack = torch.stack(mus, dim=1)       # (batch, 3, 2)
        sigma_stack = torch.stack(sigmas, dim=1)  # (batch, 3, 2)
        p = probs.unsqueeze(-1)                   # (batch, 3, 1)

        mu = (p * mu_stack).sum(dim=1)  # (batch, 2)
        var_combined = (p * (sigma_stack ** 2 + mu_stack ** 2)).sum(dim=1) - mu ** 2
        sigma = torch.sqrt(var_combined.clamp(min=1e-8))

        return mu, sigma, probs, d_pred
