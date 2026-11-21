"""Clean control for P7 v3: a small MLP trained only on data inside each training fold (pre-2018 in H2),
so it cannot have seen the evaluation years. Features known at close t; targets use t+1..t+h only."""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

WINDOWS = [1, 5, 10, 21, 42, 63, 126, 189, 252]


def _sd(lr: pd.DataFrame, com: int) -> pd.DataFrame:
    return lr.ewm(com=com, min_periods=com).std()


def features(R: pd.DataFrame, com: int = 60) -> pd.DataFrame:
    lr = np.log1p(R)
    sd = _sd(lr, com)
    cols = {f"m{w}": (lr.rolling(w, min_periods=w).sum() / (sd * np.sqrt(w))).clip(-5, 5) for w in WINDOWS}
    cols["logvol"] = np.log(sd * np.sqrt(252))
    F = pd.concat({k: v.stack(future_stack=True) for k, v in cols.items()}, axis=1)
    F.index.names = ["date", "inst"]
    return F


def targets(R: pd.DataFrame, horizons: dict[str, int], com: int = 60) -> pd.DataFrame:
    lr = np.log1p(R)
    sd = _sd(lr, com)
    out = {}
    for name, h in horizons.items():
        fwd = lr[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)   # sum of t+1..t+h
        out[name] = (fwd / (sd * np.sqrt(h))).clip(-3, 3).stack(future_stack=True)
    Y = pd.concat(out, axis=1)
    Y.index.names = ["date", "inst"]
    return Y


class CleanNet(nn.Module):
    def __init__(self, n_in: int, n_out: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, 32), nn.ReLU(), nn.Dropout(0.1),
                                 nn.Linear(32, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, n_out))

    def forward(self, x):
        return self.net(x)


def fit_predict(X_train: pd.DataFrame, Y_train: pd.DataFrame, X_test: pd.DataFrame, seed: int = 0,
                epochs: int = 20, device: str | None = None) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed); np.random.seed(seed)
    idx = X_train.index.intersection(Y_train.index)
    xt = torch.tensor(X_train.loc[idx].to_numpy(np.float32), device=device)
    yt = torch.tensor(Y_train.loc[idx].to_numpy(np.float32), device=device)
    mu, sd = xt.mean(0), xt.std(0) + 1e-6
    net = CleanNet(xt.shape[1], yt.shape[1]).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    g = torch.Generator(device="cpu").manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(xt), generator=g).to(device)
        for i in range(0, len(xt), 4096):
            b = perm[i:i + 4096]
            loss = ((net((xt[b] - mu) / sd) - yt[b]) ** 2).mean()
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    with torch.no_grad():
        scale = net((xt - mu) / sd).std(0) + 1e-6
        xs = torch.tensor(X_test.to_numpy(np.float32), device=device)
        p = torch.tanh(net((xs - mu) / sd) / scale).cpu().numpy()
    return pd.DataFrame(p, index=X_test.index, columns=Y_train.columns)


def clean_signal(pred: pd.DataFrame) -> pd.DataFrame:
    return pred.mean(axis=1).unstack("inst")
