"""
attention_fusion.py  —  Attention-based multimodal fusion 

Implements two fusion strategies from the project spec:
  1. Modality Attention   : learns which modality to trust more per sample
  2. Cross-modal Attention: lets each modality attend to others (transformer-style)

Architecture:
  Input modalities → Modality Attention → Weighted sum → LSTM → Prediction
                         ↑
                  Cross-modal Attention (optional, richer)

"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ── Modality dimensions (must match features.py build_feature_vector) ─────────
MODALITY_DIMS = {
    "weather":  6,
    "cloud":    512,
    "thermal":  512,
    "text":     384,
    "forecast": 10,
    "geo":      5,
    "radar":    4,
    "nwp":      4,
}
MODALITY_NAMES = list(MODALITY_DIMS.keys())
TOTAL_DIM      = sum(MODALITY_DIMS.values())   # 1437


def split_modalities(X: "np.ndarray") -> dict:
    """
    Split a flat 1437-d feature vector back into per-modality blocks.
    Returns dict {modality_name: tensor (batch, dim)}.
    """
    if not _TORCH_OK:
        return {}

    splits, idx = {}, 0
    for name, dim in MODALITY_DIMS.items():
        splits[name] = torch.tensor(X[:, idx:idx+dim], dtype=torch.float32)
        idx += dim
    return splits


class ModalityAttentionFusion(nn.Module):
    """
    Soft attention over modalities.

    For each sample, learns a scalar weight per modality (via a small MLP),
    then computes a weighted sum of all modality embeddings projected to
    a common hidden_dim.

    Steps:
      1. Project each modality → hidden_dim  (learned linear)
      2. Compute attention score per modality  (MLP over projected repr.)
      3. Softmax over 8 modality scores  → attention weights
      4. Weighted sum of projected modalities → fused_dim output
    """
    def __init__(self, hidden_dim=128, fused_dim=256):
        super().__init__()
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim)
            for name, dim in MODALITY_DIMS.items()
        })
        # Attention scorer: hidden_dim → scalar
        self.attn_scorer = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )
        self.output_proj = nn.Linear(hidden_dim, fused_dim)
        self.norm        = nn.LayerNorm(fused_dim)

    def forward(self, modality_dict: dict):
        """
        modality_dict: {name: tensor (batch, modality_dim)}
        Returns: fused tensor (batch, fused_dim)
        """
        projected = []
        for name in MODALITY_NAMES:
            x = modality_dict[name]                  # (B, modality_dim)
            h = torch.relu(self.projections[name](x))# (B, hidden_dim)
            projected.append(h)

        stacked = torch.stack(projected, dim=1)      # (B, n_mod, hidden_dim)

        # Attention scores
        scores = self.attn_scorer(stacked)            # (B, n_mod, 1)
        weights = F.softmax(scores, dim=1)            # (B, n_mod, 1)  sum=1

        # Weighted sum
        fused = (stacked * weights).sum(dim=1)        # (B, hidden_dim)
        return self.norm(self.output_proj(fused))      # (B, fused_dim)

    def get_attention_weights(self, modality_dict: dict):
        """Return attention weights per modality for interpretability."""
        projected = [
            torch.relu(self.projections[name](modality_dict[name]))
            for name in MODALITY_NAMES
        ]
        stacked = torch.stack(projected, dim=1)
        scores  = self.attn_scorer(stacked)
        weights = F.softmax(scores, dim=1).squeeze(-1)  # (B, n_mod)
        return {name: weights[:, i].detach().cpu().numpy()
                for i, name in enumerate(MODALITY_NAMES)}


class CrossModalAttentionFusion(nn.Module):
    """
    Transformer-style cross-modal attention.
    Each modality attends to every other modality (query-key-value).

    Steps:
      1. Project all modalities to hidden_dim
      2. Stack as sequence: (batch, n_modalities, hidden_dim)
      3. Apply multi-head self-attention  (modalities attend to each other)
      4. Mean-pool attended representations → fused output

    This is the "multimodal transformer" architecture from the spec.
    """
    def __init__(self, hidden_dim=128, n_heads=4, fused_dim=256):
        super().__init__()
        self.projections = nn.ModuleDict({
            name: nn.Linear(dim, hidden_dim)
            for name, dim in MODALITY_DIMS.items()
        })
        self.cross_attn  = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.1,
        )
        self.ff      = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm1       = nn.LayerNorm(hidden_dim)
        self.norm2       = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, fused_dim)

    def forward(self, modality_dict: dict):
        projected = torch.stack([
            torch.relu(self.projections[name](modality_dict[name]))
            for name in MODALITY_NAMES
        ], dim=1)                                       # (B, n_mod, hidden_dim)

        # Self-attention across modalities
        attn_out, _ = self.cross_attn(projected, projected, projected)
        x = self.norm1(projected + attn_out)            # residual
        x = self.norm2(x + self.ff(x))                  # FFN + residual
        fused = x.mean(dim=1)                           # (B, hidden_dim)
        return self.output_proj(fused)                  # (B, fused_dim)


class CNNLSTMAttentionModel(nn.Module):
    """
    Full CNN + LSTM + Attention hybrid (Step 7 deep learning model).

    Architecture:
      Flat feature vector
        → split into 8 modality blocks
        → CrossModalAttentionFusion  (modalities attend to each other)
        → reshape as sequence  (seq_len, fused_dim)
        → BiLSTM                     (temporal modeling)
        → Dropout
        → Linear → Sigmoid           (binary event prediction)

    This combines:
      - CNN representations  (ResNet18 features are in cloud/thermal blocks)
      - Transformer attention (cross-modal attention)
      - LSTM temporal modeling
      All three architectures the spec requests.
    """
    def __init__(self, seq_len=1, hidden=128, fused_dim=256, lstm_layers=2):
        super().__init__()
        self.seq_len   = seq_len
        self.fused_dim = fused_dim

        self.fusion = CrossModalAttentionFusion(
            hidden_dim=hidden,
            n_heads=4,
            fused_dim=fused_dim,
        )
        self.lstm = nn.LSTM(
            input_size=fused_dim,
            hidden_size=hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.dropout = nn.Dropout(0.4)
        self.fc      = nn.Linear(hidden * 2, 1)   # *2 for bidirectional
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_flat):
        """
        x_flat: (batch, 1437) — flat feature vector
        Returns: (batch, 1) probability
        """
        mods   = split_modalities(x_flat.detach().cpu().numpy()
                                  if not isinstance(x_flat, dict) else x_flat)
        # Move modalities to same device as model
        device = next(self.parameters()).device
        mods   = {k: v.to(device) for k, v in mods.items()}

        fused  = self.fusion(mods)                    # (B, fused_dim)
        seq    = fused.unsqueeze(1)                   # (B, 1, fused_dim)
        out, _ = self.lstm(seq)                       # (B, 1, hidden*2)
        out    = self.dropout(out[:, -1, :])          # (B, hidden*2)
        return self.sigmoid(self.fc(out))             # (B, 1)


# ── Training helper ───────────────────────────────────────────────────────────

def train_attention_model(X, y, event, epochs=30, hidden=128, fused_dim=256):
    """
    Train CNNLSTMAttentionModel for one event.
    Saves to models/attn_{event}.pt
    Returns val_f1 score.
    """
    if not _TORCH_OK:
        print(f"[Attention] PyTorch unavailable — skipping {event}")
        return None

    import torch
    from torch.utils.data import TensorDataset, DataLoader
    from sklearn.metrics import f1_score as sk_f1

    n      = len(X)
    split  = int(n * 0.8)
    if split < 10:
        print(f"[Attention] Not enough data for {event}")
        return None

    Xt = torch.tensor(X[:split], dtype=torch.float32)
    yt = torch.tensor(y[:split], dtype=torch.float32).unsqueeze(1)
    Xv = torch.tensor(X[split:], dtype=torch.float32)
    yv = y[split:]

    loader  = DataLoader(TensorDataset(Xt, yt), batch_size=32, shuffle=False)
    model   = CNNLSTMAttentionModel(hidden=hidden, fused_dim=fused_dim)
    opt     = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
    loss_fn = torch.nn.BCELoss()

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        if (ep + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                probs = model(Xv).squeeze().numpy()
            preds = (probs > 0.5).astype(int)
            f1 = sk_f1(yv, preds, zero_division=0)
            print(f"  [Attention/{event}] Epoch {ep+1}/{epochs} — val F1={f1:.3f}")
            model.train()

    model.eval()
    with torch.no_grad():
        probs = model(Xv).squeeze().numpy()
    preds  = (probs > 0.5).astype(int)
    val_f1 = sk_f1(yv, preds, zero_division=0)

    torch.save({
        "state_dict": model.state_dict(),
        "hidden":     hidden,
        "fused_dim":  fused_dim,
    }, f"models/attn_{event}.pt")
    print(f"  [Attention/{event}] Saved → models/attn_{event}.pt  F1={val_f1:.3f}")
    return val_f1


def load_attention_models():
    """
    Load all saved attention models. Returns {event: model} or {}.
    """
    if not _TORCH_OK:
        return {}

    models = {}
    for event in ["rain", "heat", "wind", "snow", "haze"]:
        path = f"models/attn_{event}.pt"
        if not __import__("os").path.exists(path):
            continue
        try:
            ckpt  = torch.load(path, map_location="cpu", weights_only=False)
            model = CNNLSTMAttentionModel(
                hidden   =ckpt.get("hidden",    128),
                fused_dim=ckpt.get("fused_dim", 256),
            )
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            models[event] = model
            print(f"[Attention] Loaded model for {event}")
        except Exception as e:
            print(f"[Attention] Could not load {path}: {e}")
    return models


def predict_attention(models: dict, X: np.ndarray) -> dict:
    """
    Run inference with attention models.
    X: (1, 1437) numpy array
    Returns {event: {"detected": bool, "confidence": float, "attn_weights": dict}}
    """
    if not _TORCH_OK or not models:
        return {}

    results = {}
    x_t = torch.tensor(X, dtype=torch.float32)

    for event, model in models.items():
        with torch.no_grad():
            prob = model(x_t).item()
        # Get attention weights for interpretability
        mods  = split_modalities(X)
        attn  = {}
        if hasattr(model.fusion, "get_attention_weights"):
            try:
                attn = model.fusion.get_attention_weights(mods)
                attn = {k: float(v[0]) for k, v in attn.items()}
            except Exception:
                pass
        results[event] = {
            "detected":    prob > 0.5,
            "confidence":  round(prob, 4),
            "attn_weights": attn,
        }
    return results
