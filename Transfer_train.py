# transfer_dw_pigru.py
# ------------------------------------------------------------
# Transfer learning for DoubleNash PI-GRU/LSTM to DW
# Handles different rainfall station counts by:
#   1) Areal precipitation aggregation (mean of DW stations)
#   2) Repeat the areal rain to match the source model rain-dimension
# Refit scalers on DW data (must), reuse pretrained weights (optional).
# Default transfer: freeze GRU backbone, finetune physics params + fc_out.
# ------------------------------------------------------------

import os
import math
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


# ---- Import your model & loss from your latest code ----
# Make sure DoubleNashPIGRU_SHAP.py is in the same folder or in PYTHONPATH.
from DoubleNashPIGRU_SHAP import PureGRUModel, PhysicsInformedLoss


# -------------------------
# Metrics
# -------------------------
def nse(sim, obs, eps=1e-12):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    den = np.sum((obs - np.mean(obs)) ** 2) + eps
    return 1.0 - np.sum((sim - obs) ** 2) / den


def kge(sim, obs, eps=1e-12):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)

    r = np.corrcoef(sim, obs)[0, 1]
    if np.isnan(r):
        r = 0.0
    beta = (np.mean(sim) + eps) / (np.mean(obs) + eps)
    gamma = (np.std(sim) + eps) / (np.std(obs) + eps)
    return 1.0 - np.sqrt((r - 1.0) ** 2 + (beta - 1.0) ** 2 + (gamma - 1.0) ** 2)


def rmse(sim, obs):
    sim = np.asarray(sim, dtype=float)
    obs = np.asarray(obs, dtype=float)
    return float(np.sqrt(np.mean((sim - obs) ** 2)))


# -------------------------
# Dataset (sliding window)
# -------------------------
class SeqDataset(Dataset):
    def __init__(self, data_tensor, window_size, pre_len):
        """
        data_tensor: torch.FloatTensor [N, F] where last column is y_norm(logQ scaled)
        """
        self.x = data_tensor
        self.window = window_size
        self.pre_len = pre_len
        self.N = data_tensor.shape[0]

        self.max_i = self.N - self.window - self.pre_len + 1
        if self.max_i <= 0:
            raise ValueError(f"Data too short: N={self.N}, window={self.window}, pre_len={self.pre_len}")

    def __len__(self):
        return self.max_i

    def __getitem__(self, idx):
        seq = self.x[idx: idx + self.window]                      # [window, F]
        label = self.x[idx + self.window: idx + self.window + self.pre_len, -1:]  # [pre_len, 1]
        return seq, label


# -------------------------
# Utilities
# -------------------------

def infer_rnn_type_from_ckpt(model_ckpt_path: str):
    """
    Infer whether the pretrained model is GRU or LSTM from state_dict keys.
    """
    sd = torch.load(model_ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    for k in sd.keys():
        if k.startswith("gru.") or "gru." in k:
            return "GRU"
        if k.startswith("lstm.") or "lstm." in k:
            return "LSTM"

    raise RuntimeError("Cannot infer RNN type from checkpoint (no GRU/LSTM keys found)")



def infer_input_size_from_ckpt(model_ckpt_path: str):
    """
    Infer RNN input_size from pretrained checkpoint.
    Supports both GRU and LSTM.

    GRU:  weight_ih_l0 shape = (3*hidden_size, input_size)
    LSTM: weight_ih_l0 shape = (4*hidden_size, input_size)
    """
    sd = torch.load(model_ckpt_path, map_location="cpu")

    # state_dict could be wrapped
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]

    key = None
    rnn_type = None

    # 1) Try GRU
    for k in sd.keys():
        if "gru.weight_ih_l0" in k:
            key = k
            rnn_type = "GRU"
            break

    # 2) Try LSTM
    if key is None:
        for k in sd.keys():
            if "lstm.weight_ih_l0" in k:
                key = k
                rnn_type = "LSTM"
                break

    if key is None:
        raise RuntimeError(
            "Cannot infer input_size: neither GRU nor LSTM weight_ih_l0 found in checkpoint"
        )

    w = sd[key]
    input_size = w.shape[1]

    print(f"[Info] Detected source RNN type: {rnn_type}, input_size={input_size}")

    return int(input_size)



def build_dw_dataframe(dw_csv):
    df = pd.read_csv(dw_csv)

    # Identify time and obs columns
    time_col_candidates = ["Time", "tm", "time", "datetime", "date"]
    time_col = None
    for c in time_col_candidates:
        if c in df.columns:
            time_col = c
            break

    if "OBS" not in df.columns:
        raise ValueError("DW_flood.csv must contain column 'OBS'")

    rain_cols = [c for c in df.columns if c not in ([time_col] if time_col else []) + ["OBS"]]
    if len(rain_cols) == 0:
        raise ValueError("No rainfall columns found in DW_flood.csv")

    return df, time_col, rain_cols


def make_aligned_features(df, time_col, rain_cols, src_rain_dim):
    """
    1) areal rain = mean of DW station rain
    2) repeat areal rain to match src_rain_dim (so that pretrained GRU input size matches)
    3) y = OBS
    """
    rain = df[rain_cols].to_numpy(dtype=float)  # [N, n_dw]
    obs = df["OBS"].to_numpy(dtype=float).reshape(-1, 1)  # [N,1]

    # areal precipitation (simple mean; you can replace by Thiessen weights if available)
    rain_mean = np.mean(rain, axis=1, keepdims=True)  # [N,1]

    # repeat to src_rain_dim
    rain_rep = np.repeat(rain_mean, repeats=src_rain_dim, axis=1)  # [N, src_rain_dim]

    # return raw arrays
    t = df[time_col].to_numpy() if time_col else np.arange(len(df))
    return t, rain_rep, obs


def fit_scalers_and_pack(rain_rep, obs, train_ratio=0.6, valid_ratio=0.2):
    """
    Fit scaler_x on rainfall features, scaler_y on log1p(obs).
    Build packed tensor [X_norm, y_norm] where y_norm is last column.
    Split chronologically.
    """
    N = len(obs)
    n_train = int(N * train_ratio)
    n_valid = int(N * valid_ratio)
    n_test = N - n_train - n_valid
    if n_test <= 0:
        raise ValueError("Split ratios lead to empty test set. Adjust train/valid ratios.")

    X = rain_rep
    y = obs

    X_train, X_valid, X_test = X[:n_train], X[n_train:n_train+n_valid], X[n_train+n_valid:]
    y_train, y_valid, y_test = y[:n_train], y[n_train:n_train+n_valid], y[n_train+n_valid:]

    scaler_x = StandardScaler()
    X_train_n = scaler_x.fit_transform(X_train)
    X_valid_n = scaler_x.transform(X_valid)
    X_test_n  = scaler_x.transform(X_test)

    scaler_y = StandardScaler()
    y_train_log = np.log1p(np.maximum(y_train, 0.0))
    y_valid_log = np.log1p(np.maximum(y_valid, 0.0))
    y_test_log  = np.log1p(np.maximum(y_test, 0.0))

    y_train_n = scaler_y.fit_transform(y_train_log)
    y_valid_n = scaler_y.transform(y_valid_log)
    y_test_n  = scaler_y.transform(y_test_log)

    # Packed: [X_norm, y_norm]  (y_norm as last col to serve as OBS_HIS)
    train_pack = np.hstack([X_train_n, y_train_n])
    valid_pack = np.hstack([X_valid_n, y_valid_n])
    test_pack  = np.hstack([X_test_n,  y_test_n])

    splits = {
        "n_train": n_train,
        "n_valid": n_valid,
        "n_test": n_test
    }

    return (train_pack, valid_pack, test_pack), (y_train, y_valid, y_test), (scaler_x, scaler_y), splits


def inverse_y(y_pred_norm, scaler_y):
    """
    y_pred_norm: torch.Tensor [B, pre_len] or [B, pre_len, 1]
    returns discharge in original unit (m3/s) as torch.Tensor [B, pre_len]
    """
    if y_pred_norm.dim() == 3:
        y_pred_norm = y_pred_norm.squeeze(-1)

    y_scale = torch.tensor(scaler_y.scale_, device=y_pred_norm.device, dtype=y_pred_norm.dtype)
    y_mean  = torch.tensor(scaler_y.mean_,  device=y_pred_norm.device, dtype=y_pred_norm.dtype)

    y_log = y_pred_norm * y_scale + y_mean
    q = torch.relu(torch.expm1(y_log))
    return q


# -------------------------
# Training / Evaluation
# -------------------------
@torch.no_grad()
def eval_epoch(model, criterion, loader, scaler_y, device):
    model.eval()
    criterion.eval()

    all_obs = []
    all_sim = []

    for seq, labels in loader:
        seq = seq.to(device)          # [B, W, F]
        labels = labels.to(device)    # [B, pre_len, 1]

        # Replace last feature (OBS_HIS) with phys simulated feature q_sim_norm (optional)
        q_sim_norm, _ = criterion.compute_simulated_feature(seq)
        seq_in = torch.cat([seq[:, :, :-1], q_sim_norm], dim=2)

        y_pred = model(seq_in)  # [B, pre_len]
        # Convert to discharge
        q_pred = inverse_y(y_pred, scaler_y)      # [B, pre_len]
        q_true = inverse_y(labels.squeeze(-1), scaler_y)

        # only evaluate first lead (pre_len can be >1)
        all_obs.append(q_true[:, 0].detach().cpu().numpy())
        all_sim.append(q_pred[:, 0].detach().cpu().numpy())

    obs = np.concatenate(all_obs)
    sim = np.concatenate(all_sim)

    return {
        "NSE": nse(sim, obs),
        "KGE": kge(sim, obs),
        "RMSE": rmse(sim, obs),
        "obs": obs,
        "sim": sim
    }


def train_transfer(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Load DW data ----
    df, time_col, rain_cols = build_dw_dataframe(args.dw_csv)

    # ---- Determine source input size (rain_dim + 1) ----
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        src_input_size = infer_input_size_from_ckpt(args.pretrained_model)
        src_rain_dim = src_input_size - 1
        print(f"[Info] Inferred source input_size={src_input_size}, src_rain_dim={src_rain_dim}")
    else:
        # If no checkpoint, we can train from scratch using DW station count (or aggregated).
        # We still recommend using areal mean + repeat to a chosen dim for stability.
        src_rain_dim = args.src_rain_dim
        src_input_size = src_rain_dim + 1
        print(f"[Info] No pretrained model. Using src_rain_dim={src_rain_dim}, input_size={src_input_size}")

    # ---- Build aligned features ----
    t, rain_rep, obs = make_aligned_features(df, time_col, rain_cols, src_rain_dim)

    # ---- Fit scalers on DW and pack ----
    (train_pack, valid_pack, test_pack), (y_tr, y_va, y_te), (scaler_x, scaler_y), splits = \
        fit_scalers_and_pack(rain_rep, obs, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio)

    print(f"[Split] train={splits['n_train']} valid={splits['n_valid']} test={splits['n_test']}")
    print(f"[DW] rain stations={len(rain_cols)} -> areal mean -> repeated to {src_rain_dim} dims")

    # ---- Build loaders ----
    train_tensor = torch.tensor(train_pack, dtype=torch.float32)
    valid_tensor = torch.tensor(valid_pack, dtype=torch.float32)
    test_tensor  = torch.tensor(test_pack,  dtype=torch.float32)

    train_ds = SeqDataset(train_tensor, args.window, args.pre_len)
    valid_ds = SeqDataset(valid_tensor, args.window, args.pre_len)
    test_ds  = SeqDataset(test_tensor,  args.window, args.pre_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch, shuffle=False)

    # ---- Init model & criterion ----
    # ===== 自动识别 RNN 类型 =====
    rnn_type = infer_rnn_type_from_ckpt(args.pretrained_model)
    print(f"[Info] Detected source RNN type: {rnn_type}")

    if rnn_type == "GRU":
        from DoubleNashPIGRU_SHAP import PureGRUModel as RNNModel
    elif rnn_type == "LSTM":
        from DoubleNashPILSTM import PureLSTMModel as RNNModel
    else:
        raise RuntimeError(f"Unsupported RNN type: {rnn_type}")

    model = RNNModel(
        input_size=src_input_size,
        output_size=1,
        hidden_size=args.hidden,
        num_layers=args.layers,
        pre_len=args.pre_len,
        dropout=args.dropout
    ).to(device)

    # Use uniform station weights with length=src_rain_dim (so compute_simulated_feature uses weighted sum)
    station_weights = [1.0 / src_rain_dim] * src_rain_dim

    criterion = PhysicsInformedLoss(
        area_km2=args.area_km2,
        station_weights=station_weights,
        dt_hours=1.0,
        weight_mse=args.w_mse,
        weight_phys=args.w_phys,
        w_log=args.w_log,
        device=str(device),
        scaler_x=scaler_x,
        scaler_y=scaler_y
    ).to(device)

    # ---- Load pretrained weights (optional) ----
    if args.pretrained_model and os.path.exists(args.pretrained_model):
        sd = torch.load(args.pretrained_model, map_location=device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
        model.load_state_dict(sd, strict=True)
        print(f"[Load] pretrained model: {args.pretrained_model}")

    if args.pretrained_criterion and os.path.exists(args.pretrained_criterion):
        cd = torch.load(args.pretrained_criterion, map_location=device)
        if isinstance(cd, dict) and "state_dict" in cd:
            cd = cd["state_dict"]
        # scaler-related buffers differ -> allow non-strict
        criterion.load_state_dict(cd, strict=False)
        print(f"[Load] pretrained criterion: {args.pretrained_criterion}")

        if hasattr(criterion, "state_slow"):
            criterion.state_slow = criterion.state_slow * 0.1
        if hasattr(criterion, "state_fast"):
            criterion.state_fast = criterion.state_fast * 0.5

        print("[Transfer] Physical states damped for target basin (slow flow reset).")

        area_ratio = 0.56
        k_scale = area_ratio ** 0.55  # ≈ 0.77
        n_scale = 0.9  # 经验值，可 0.8~0.9

        for name, p in criterion.named_parameters():
            if ("k" in name.lower()) or ("n" in name.lower()):
                p.requires_grad = False

        with torch.no_grad():
            for name, p in criterion.named_parameters():
                lname = name.lower()
                if "k" in lname:
                    p.mul_(k_scale)
                if "n" in lname:
                    p.mul_(n_scale)

        print(f"[Transfer] IUH parameters scaled: k x {k_scale:.2f}, n x {n_scale:.2f}")

    # ---- Transfer mode: partial unfreeze (Scheme 2) ----
    if args.freeze_backbone:
        # 1) Freeze all
        for p in model.parameters():
            p.requires_grad = False

        # 2) warmup GRU
        for p in model.gru.parameters():
            p.requires_grad = True

        # 3) warmup output layer
        for p in model.fc_out.parameters():
            p.requires_grad = True

        print("[Transfer] Freeze input representation, finetune GRU + fc_out + physics params.")
    else:
        print("[Transfer] Finetune full model + physics params.")

    # freeze slow flow paras
    for name, p in criterion.named_parameters():
        if ("slow" in name.lower()) or ("k2" in name.lower()) or ("n2" in name.lower()):
            p.requires_grad = False

    print("[Transfer] Slow-flow physical parameters frozen.")

    # Set mask prob (anti-lazy) if you use it
    model.mask_prob = args.mask_prob

    # ---- Optimizer: physics params + trainable model params ----
    params = []
    # params += [p for p in criterion.parameters() if p.requires_grad]
    params += [p for p in model.parameters() if p.requires_grad]

    params += [p for p in criterion.parameters() if p.requires_grad]
    print(f"[Optimizer] Trainable parameters: {len(params)}")
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # ---- Training loop ----
    best_score = -1e9
    best_path_model = args.out_model
    best_path_crit  = args.out_criterion

    patience = args.patience
    wait = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        criterion.train()

        epoch_loss = 0.0
        n_batches = 0

        for seq, labels in train_loader:
            seq = seq.to(device)          # [B, W, F]
            labels = labels.to(device)    # [B, pre_len, 1]

            # Replace OBS_HIS with phys simulated feature
            q_sim_norm, _ = criterion.compute_simulated_feature(seq)
            seq_in = torch.cat([seq[:, :, :-1], q_sim_norm], dim=2)

            y_pred = model(seq_in)  # [B, pre_len]

            loss = criterion(y_pred, labels, seq, mode="combined")

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())
            n_batches += 1

        epoch_loss /= max(n_batches, 1)

        # Validate
        val = eval_epoch(model, criterion, valid_loader, scaler_y, device)
        score = val["KGE"]  # choose KGE as primary for stability
        print(f"Epoch {epoch:03d} | loss={epoch_loss:.5f} | "
              f"Val NSE={val['NSE']:.3f} KGE={val['KGE']:.3f} RMSE={val['RMSE']:.3f}")

        # Early stopping by KGE (or use NSE if you prefer)
        if score > best_score:
            best_score = score
            wait = 0
            torch.save(model.state_dict(), best_path_model)
            torch.save(criterion.state_dict(), best_path_crit)
        else:
            wait += 1
            if wait >= patience:
                print(f"[EarlyStop] No improvement in {patience} epochs.")
                break

    # ---- Test ----
    model.load_state_dict(torch.load(best_path_model, map_location=device))
    criterion.load_state_dict(torch.load(best_path_crit, map_location=device), strict=False)

    te = eval_epoch(model, criterion, test_loader, scaler_y, device)
    print(f"\n[Test] NSE={te['NSE']:.3f} KGE={te['KGE']:.3f} RMSE={te['RMSE']:.3f}")

    # ---- Export test CSV (aligned to test samples start times) ----
    # test_ds produces predictions for indices within the test segment; we map them back to time.
    # For each sample idx, prediction corresponds to time index: idx + window (first lead)
    n_train = splits["n_train"]
    n_valid = splits["n_valid"]
    test_start = n_train + n_valid

    # sample-level timestamps
    sample_times = []
    for i in range(len(test_ds)):
        ti = test_start + i + args.window  # first-step target time index
        if ti < len(t):
            sample_times.append(t[ti])
        else:
            sample_times.append(None)

    out_df = pd.DataFrame({
        "Time": sample_times,
        "OBS": te["obs"],
        "SIM": te["sim"],
        "ERR": te["sim"] - te["obs"]
    })
    out_df.to_csv(args.out_csv, index=False, encoding="utf-8-sig")
    print(f"[Save] Test predictions -> {args.out_csv}")
    print(f"[Save] Best model -> {best_path_model}")
    print(f"[Save] Best criterion -> {best_path_crit}")


def main():
    p = argparse.ArgumentParser()

    p.add_argument("--dw_csv", type=str, default=".\\data\\Random_scenarios_DW.csv", help="Path to DW_flood.csv")
    p.add_argument("--area_km2", type=float, default=2623.0, help="Target basin area (km^2)")

    # Pretrained weights (source basin)
    p.add_argument("--pretrained_model", type=str, default="best_model.pth", help="Source basin model ckpt")
    p.add_argument("--pretrained_criterion", type=str, default="best_criterion.pth", help="Source basin criterion ckpt")

    # If no pretrained model, choose a rain-dim for repeated areal rain
    p.add_argument("--src_rain_dim", type=int, default=14, help="Used only if no pretrained_model")

    # Data / sequence
    p.add_argument("--window", type=int, default=72, help="Window size (hours)")
    p.add_argument("--pre_len", type=int, default=12, help="Prediction length (steps)")

    # Split ratios (chronological)
    p.add_argument("--train_ratio", type=float, default=0.65)
    p.add_argument("--valid_ratio", type=float, default=0.2)

    # Model hyperparameters (must match your source model if loading weights)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.2)

    # Transfer strategy
    p.add_argument("--freeze_backbone", action="store_true", help="Freeze GRU, finetune fc_out + physics only")
    p.add_argument("--mask_prob", type=float, default=0.0, help="Mask prob for OBS_HIS during training (anti-lazy)")

    # Loss weights (use your current defaults or adjust)
    p.add_argument("--w_mse", type=float, default=0.5)
    p.add_argument("--w_phys", type=float, default=0.5)
    p.add_argument("--w_log", type=float, default=0.5)

    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", type=str, default="cuda")

    # Outputs
    p.add_argument("--out_model", type=str, default="dw_transfer_best_model.pth")
    p.add_argument("--out_criterion", type=str, default="dw_transfer_best_criterion.pth")
    p.add_argument("--out_csv", type=str, default="dw_transfer_test_predictions.csv")

    args = p.parse_args()
    train_transfer(args)


if __name__ == "__main__":
    main()
