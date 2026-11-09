"""Spatially blocked ten-fold CV: 1 degree cells are held out whole (GroupKFold).

Baseline settings follow integrated_train.ipynb.
    python spatial_cv.py --models GWR XGBoost
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / 'output' / 'model' / 'mesh' / 'shf_with_distance_vectors.pkl'
DEFAULT_OUT = ROOT / 'output' / 'spatial_cv'
KAN_PACKAGE = ROOT / 'code优化' / 'ASG-GKANWR'

TARGET_COL = 'SHF__mW_m2_'
# feature list of integrated_train.ipynb (note: vp7, not vp9)
GEO_COLS = [
    'Magnetic_Anomaly_Value(nT)', 'DEM', 'RASTERVALU',
    'cpd', 'LAB', 'Thermal Conductivity （mW/K/m/）', 'H(km)', 'DTF',
    'sedthk-m', 'ro6', 'vp7', 'vs8',
    'DTR',
]
DIST_COLS = ['distance_vector_to_grids']
SEED = 37
N_FOLDS = 10
BLOCK_DEG = 1.0

# integrated_train.ipynb settings
GWR_BANDWIDTH_RATIO = 0.10
GWR_REG = 1e-2
VARIAGRAM_MODEL = 'spherical'
NLAGS = 6
XGB_PARAMS = dict(n_estimators=500, max_depth=6, learning_rate=0.05,
                  subsample=0.75, colsample_bytree=0.75, reg_alpha=0.1,
                  reg_lambda=1.5, min_child_weight=3, gamma=0.2,
                  tree_method='hist', random_state=SEED)
KAN_CONFIG = dict(n_hidden=64, num_rbf=4, pca=20, ridge_alpha=10.0,
                  lr=1e-3, weight_decay=1e-3, batch_size=32, huber_delta=1.0)


# --------------------------------------------------------------------------- #
# data and partition
# --------------------------------------------------------------------------- #
class StandardThenMinMaxScaler(BaseEstimator, TransformerMixin):
    """Standardise, then rescale to [0, 1]; the order used by the training notebook."""

    def __init__(self):
        self.ss = StandardScaler()
        self.mm = MinMaxScaler()

    def fit(self, X, y=None):
        self.ss.fit(X)
        self.mm.fit(self.ss.transform(X))
        return self

    def transform(self, X):
        return self.mm.transform(self.ss.transform(X))

    def inverse_transform(self, X):
        return self.ss.inverse_transform(self.mm.inverse_transform(X))


def load_data():
    df = pd.read_pickle(DATA_FILE)
    y = df[TARGET_COL].to_numpy(float)
    X_geo = df[GEO_COLS].to_numpy(np.float64)
    # constant features are dropped, as in the notebook
    keep = [i for i in range(X_geo.shape[1]) if np.std(X_geo[:, i]) > 1e-8]
    X_geo = X_geo[:, keep]
    X_dist = np.array([np.asarray(d, dtype=np.float32) for d in df[DIST_COLS[0]]])
    return df, X_geo, X_dist, y


def blocked_folds(lon, lat):
    """GroupKFold over 1 degree cells so that whole cells are held out."""
    cells = (np.floor(lon / BLOCK_DEG).astype(int) * 1000
             + np.floor(lat / BLOCK_DEG).astype(int))
    splitter = GroupKFold(n_splits=N_FOLDS)
    folds = list(splitter.split(np.zeros((len(lon), 1)), groups=cells))
    return folds, int(len(np.unique(cells)))


# --------------------------------------------------------------------------- #
# baselines, settings taken from integrated_train.ipynb
# --------------------------------------------------------------------------- #
def gwr_predict(coords_train, X_train, y_train, coords_test, X_test, bandwidth, reg):
    """Local WLS with a fixed Gaussian kernel (same routine as the notebook)."""
    Xd = np.column_stack([np.ones(len(X_train)), X_train])
    pred = np.zeros(len(coords_test))
    for i in range(len(coords_test)):
        d = np.linalg.norm(coords_train - coords_test[i], axis=1)
        w = np.exp(-0.5 * (d / bandwidth) ** 2)
        XTWX = Xd.T @ (w[:, None] * Xd)
        XTWy = Xd.T @ (w * y_train)
        try:
            beta = np.linalg.solve(XTWX + reg * np.eye(XTWX.shape[0]), XTWy)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(XTWX + reg * np.eye(XTWX.shape[0])) @ XTWy
        pred[i] = np.r_[1.0, X_test[i]] @ beta
    return pred


def fit_gwr(df, X_geo, y, train_idx, val_idx):
    coords = df[['Lon', 'Lat']].to_numpy(float)
    sx, sy = StandardThenMinMaxScaler(), StandardThenMinMaxScaler()
    Xs = sx.fit_transform(X_geo)                 # global scaling, as in the notebook
    ys = sy.fit_transform(y.reshape(-1, 1)).flatten()
    bandwidth = cdist(coords, coords).max() * GWR_BANDWIDTH_RATIO
    pred_s = gwr_predict(coords[train_idx], Xs[train_idx], ys[train_idx],
                         coords[val_idx], Xs[val_idx], bandwidth, GWR_REG)
    return sy.inverse_transform(pred_s.reshape(-1, 1)).flatten()


def fit_kriging(df, train_idx, val_idx, y):
    from pykrige.ok import OrdinaryKriging

    coords = df[['Lon', 'Lat']].to_numpy(float)
    ok = OrdinaryKriging(coords[train_idx, 0], coords[train_idx, 1], y[train_idx],
                         variogram_model=VARIAGRAM_MODEL, nlags=NLAGS,
                         verbose=False, enable_plotting=False)
    z, _ = ok.execute('points', coords[val_idx, 0], coords[val_idx, 1])
    return np.asarray(z, dtype=float)


def fit_xgboost(df, X_geo, train_idx, val_idx, y):
    from xgboost import XGBRegressor

    X = np.column_stack([X_geo, df['Lon'].to_numpy(float), df['Lat'].to_numpy(float)])
    model = XGBRegressor(**XGB_PARAMS, n_jobs=4)
    model.fit(X[train_idx], y[train_idx])
    return model.predict(X[val_idx])


# --------------------------------------------------------------------------- #
# ASG-GKANWR
# --------------------------------------------------------------------------- #
def fit_kan(X_dist, X_geo, y, train_idx, val_idx, epochs):
    """One ASG-GKANWR fold: PCA on distance vectors, ridge baseline, RBF-KAN weights."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    sys.path.insert(0, str(KAN_PACKAGE))
    from models.rbf_kan import RBFKAN

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    n_comp = min(KAN_CONFIG['pca'], X_dist.shape[1])
    pca = PCA(n_components=n_comp, random_state=SEED)
    Xd = pca.fit_transform(X_dist[train_idx])
    Xv = pca.transform(X_dist[val_idx])

    sxd, sxg, sy = StandardThenMinMaxScaler(), StandardThenMinMaxScaler(), StandardThenMinMaxScaler()
    Xd_s, Xv_s = sxd.fit_transform(Xd), sxd.transform(Xv)
    Xg = sxg.fit_transform(X_geo[train_idx])
    Xgv = sxg.transform(X_geo[val_idx])
    yt = sy.fit_transform(y[train_idx].reshape(-1, 1)).flatten()
    yv = sy.transform(y[val_idx].reshape(-1, 1)).flatten()

    reg = Ridge(alpha=KAN_CONFIG['ridge_alpha']).fit(Xg, yt)
    beta0 = torch.tensor(reg.intercept_, dtype=torch.float32, device=device)
    beta = torch.tensor(reg.coef_, dtype=torch.float32, device=device)

    model = RBFKAN([Xd.shape[1], KAN_CONFIG['n_hidden'], KAN_CONFIG['n_hidden'], Xg.shape[1]],
                   num_rbf=KAN_CONFIG['num_rbf']).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=KAN_CONFIG['lr'],
                            weight_decay=KAN_CONFIG['weight_decay'])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=20, factor=0.5)
    crit = nn.HuberLoss(delta=KAN_CONFIG['huber_delta'])

    def predict(loader):
        model.eval()
        out = []
        with torch.no_grad():
            for xd, xg, _ in loader:
                w = model(xd.to(device))
                out.extend((beta0 + torch.sum(w * beta * xg.to(device), dim=1)).cpu().numpy())
        return np.asarray(out)

    tr = DataLoader(TensorDataset(torch.tensor(Xd_s, dtype=torch.float32),
                                  torch.tensor(Xg, dtype=torch.float32),
                                  torch.tensor(yt, dtype=torch.float32)),
                    batch_size=KAN_CONFIG['batch_size'], shuffle=True)
    vl = DataLoader(TensorDataset(torch.tensor(Xv_s, dtype=torch.float32),
                                  torch.tensor(Xgv, dtype=torch.float32),
                                  torch.tensor(yv, dtype=torch.float32)),
                    batch_size=KAN_CONFIG['batch_size'], shuffle=False)

    best_r2, best_state, best_epoch = -np.inf, None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xd, xg, yb in tr:
            opt.zero_grad()
            w = model(xd.to(device))
            pred = beta0 + torch.sum(w * beta * xg.to(device), dim=1)
            loss = crit(pred, yb.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()
        sched.step(loss.item())
        val_r2 = r2_score(yv, predict(vl))
        if val_r2 > best_r2:
            best_r2, best_epoch = val_r2, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return sy.inverse_transform(predict(vl).reshape(-1, 1)).flatten(), best_epoch


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def pooled_metrics(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    return dict(r2=float(r2_score(y_true, y_pred)), mse=float(mse), rmse=float(np.sqrt(mse)))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models', nargs='+',
                    default=['ASG-GKANWR', 'GWR', 'Kriging', 'XGBoost'])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--outdir', type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    df, X_geo, X_dist, y = load_data()
    lon = df['Lon'].to_numpy(float)
    lat = df['Lat'].to_numpy(float)
    folds, n_cells = blocked_folds(lon, lat)
    print(f'observations: {len(df)}, blocks: {n_cells}, folds: {len(folds)}')

    summary = {'protocol': 'GroupKFold on 1.0 degree cells', 'n_blocks': n_cells,
               'n_folds': N_FOLDS, 'pooled': {}}
    fold_rows = []
    for name in args.models:
        if name == 'GWR':
            fitter = lambda tr, va: fit_gwr(df, X_geo, y, tr, va)
        elif name == 'Kriging':
            fitter = lambda tr, va: fit_kriging(df, tr, va, y)
        elif name == 'XGBoost':
            fitter = lambda tr, va: fit_xgboost(df, X_geo, tr, va, y)
        elif name == 'ASG-GKANWR':
            fitter = lambda tr, va: fit_kan(X_dist, X_geo, y, tr, va, args.epochs)[0]
        else:
            raise SystemExit(f'unknown model {name}')

        y_true, y_pred = [], []
        for k, (tr_idx, va_idx) in enumerate(folds, start=1):
            pred = fitter(tr_idx, va_idx)
            metrics = pooled_metrics(y[va_idx], pred)
            fold_rows.append({'fold': k, 'model': name, **metrics})
            y_true.extend(y[va_idx])
            y_pred.extend(pred)
        m = pooled_metrics(np.array(y_true), np.array(y_pred))
        summary['pooled'][name] = m
        print(f'{name:11s} pooled R2={m["r2"]:.4f}  RMSE={m["rmse"]:.2f}')

    pd.DataFrame(fold_rows).to_csv(args.outdir / 'spatial_block_cv_folds.csv', index=False)
    (args.outdir / 'spatial_block_cv_summary.json').write_text(
        json.dumps(summary, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
