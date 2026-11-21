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
    return F.replace([np.inf, -np.inf], np.nan)   # sd == 0 gives -inf logvol; callers dropna


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


def _fit(X_train: pd.DataFrame, Y_train: pd.DataFrame, seed: int, epochs: int, device: str) -> dict:
    """Train the MLP; return everything needed to predict later (state, feature mean/sd, output scale)."""
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
    return {"state": {k: v.detach().cpu() for k, v in net.state_dict().items()}, "mu": mu.cpu(), "sd": sd.cpu(),
            "scale": scale.cpu(), "cols_in": list(X_train.columns), "cols_out": list(Y_train.columns)}


def _apply(m: dict, X_test: pd.DataFrame, device: str) -> pd.DataFrame:
    net = CleanNet(len(m["cols_in"]), len(m["cols_out"])).to(device)
    net.load_state_dict(m["state"]); net.eval()
    mu, sd, scale = (m[k].to(device) for k in ("mu", "sd", "scale"))
    with torch.no_grad():
        xs = torch.tensor(X_test[m["cols_in"]].to_numpy(np.float32), device=device)
        p = torch.tanh(net((xs - mu) / sd) / scale).cpu().numpy()
    return pd.DataFrame(p, index=X_test.index, columns=m["cols_out"])


def fit_predict(X_train: pd.DataFrame, Y_train: pd.DataFrame, X_test: pd.DataFrame, seed: int = 0,
                epochs: int = 20, device: str | None = None) -> pd.DataFrame:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    return _apply(_fit(X_train, Y_train, seed, epochs, device), X_test, device)


def clean_signal(pred: pd.DataFrame) -> pd.DataFrame:
    return pred.mean(axis=1).unstack("inst")


H_DEFAULT = {"short": 21, "medium": 63, "long": 252}


def _train_xy(R: pd.DataFrame, train_end: str, horizons: dict[str, int] = H_DEFAULT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Training rows with date <= train_end whose targets are fully known by train_end: rows within
    max(horizon) trading days of train_end are dropped, so no target uses a post-train_end return."""
    dates = R.index
    end_pos = int(dates.searchsorted(pd.Timestamp(train_end), side="right")) - 1
    last_ok = dates[end_pos - max(horizons.values())]
    X, Y = features(R), targets(R, horizons)
    keep = lambda D: D[D.index.get_level_values(0) <= last_ok].dropna()
    return keep(X), keep(Y)


def _test_features(R: pd.DataFrame, train_end: str) -> pd.DataFrame:
    X = features(R)
    return X[X.index.get_level_values(0) > pd.Timestamp(train_end)].dropna()


def clean_full_pred(R: pd.DataFrame, train_end: str = "2017-12-31", seed: int = 0, epochs: int = 20) -> pd.DataFrame:
    """Per-horizon predictions (index date x inst, columns short/medium/long) for all dates after train_end."""
    Xtr, Ytr = _train_xy(R, train_end)
    return fit_predict(Xtr, Ytr, _test_features(R, train_end), seed=seed, epochs=epochs)


def clean_full_signal(R: pd.DataFrame, train_end: str = "2017-12-31", seed: int = 0, epochs: int = 20) -> pd.DataFrame:
    """Clean MLP trained once on data up to train_end ("trained pre-2018"); signal for all dates after it
    (date x inst), known at close t."""
    return clean_signal(clean_full_pred(R, train_end, seed, epochs))


def save_clean_model(path, R: pd.DataFrame, train_end: str = "2017-12-31", seed: int = 0, epochs: int = 20,
                     device: str | None = None) -> None:
    """Train exactly as clean_full_signal does and freeze the model to `path` (.pt) for retrain-free prediction."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    Xtr, Ytr = _train_xy(R, train_end)
    m = _fit(Xtr, Ytr, seed, epochs, device)
    m.update(train_end=train_end, seed=seed, epochs=epochs)
    torch.save(m, path)


def clean_predict_pred(path, R: pd.DataFrame, device: str = "cpu") -> pd.DataFrame:
    """Per-horizon predictions (date x inst by short/medium/long) from the frozen model at `path`. Never retrains."""
    m = torch.load(path, map_location="cpu", weights_only=False)
    return _apply(m, _test_features(R, m["train_end"]), device)


def clean_predict(path, R: pd.DataFrame, device: str = "cpu") -> pd.DataFrame:
    """Signal (date x inst) for dates > train_end from the frozen model at `path`. Never retrains."""
    return clean_signal(clean_predict_pred(path, R, device))
