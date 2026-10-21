"""Small CNN over stacked range-Doppler and range-time maps (one channel per map and mic)."""

import numpy as np
import torch
from torch import nn


class SmallCNN(nn.Module):
    """Three conv blocks and global average pooling, ~24k parameters."""

    def __init__(self, in_ch: int = 4, n_classes: int = 4, width: int = 16, dropout: float = 0.3):
        super().__init__()

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))

        self.features = nn.Sequential(block(in_ch, width), block(width, 2 * width), block(2 * width, 4 * width),
                                      nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(4 * width, n_classes))

    def forward(self, x):
        return self.head(self.features(x))


def n_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def augment(x: torch.Tensor, n_mics: int = 2, max_shift: int = 2, gain_db: float = 2.0,
            noise: float = 0.05) -> torch.Tensor:
    """Label-preserving augmentations for a batch of standardised maps (B, C, H, W).

    Swap the mics; mirror the Doppler/time axis (approaching and receding motion are the
    same activity); shift range by a couple of bins (the same activity at another
    distance); add a level offset and a little noise.
    """
    b = x.shape[0]
    x = x.clone()
    swap = torch.rand(b) < 0.5
    if swap.any():
        order = [(i + 1) % n_mics + (i // n_mics) * n_mics for i in range(x.shape[1])]
        x[swap] = x[swap][:, order]
    flip = torch.rand(b) < 0.5
    if flip.any():
        x[flip] = torch.flip(x[flip], dims=[2])
    shifts = torch.randint(-max_shift, max_shift + 1, (b,))
    for i in torch.nonzero(shifts).flatten().tolist():
        s = int(shifts[i])
        x[i] = torch.roll(x[i], s, dims=2)          # x[i] is (C, H, W): range is dim 2
        if s > 0:
            x[i, :, :, :s] = x[i, :, :, s:s + 1]
        else:
            x[i, :, :, s:] = x[i, :, :, s - 1:s]
    x = x + gain_db * (torch.rand(b, 1, 1, 1) * 2 - 1) / 10.0
    return x + noise * torch.randn_like(x)


def train_cnn(x_train: np.ndarray, y_train: np.ndarray, n_classes: int, seed: int = 0, epochs: int = 30,
              batch: int = 32, lr: float = 1e-3, weight_decay: float = 1e-4):
    """Train on standardised maps with class-weighted cross-entropy and a fixed schedule
    (no early stopping, so the test session never influences training)."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = SmallCNN(in_ch=x_train.shape[1], n_classes=n_classes)
    counts = np.bincount(y_train, minlength=n_classes).astype(float)
    weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    xt = torch.tensor(x_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.long)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        for idx in torch.randperm(len(xt), generator=gen).split(batch):
            opt.zero_grad()
            loss = loss_fn(model(augment(xt[idx])), yt[idx])
            loss.backward()
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def predict(model: nn.Module, x: np.ndarray, batch: int = 256) -> np.ndarray:
    model.eval()
    xt = torch.tensor(x, dtype=torch.float32)
    return np.concatenate([model(xt[i:i + batch]).argmax(1).numpy() for i in range(0, len(xt), batch)])
