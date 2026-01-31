# ============================================================
# Transfer_train_DD.py
# Pure Data-Driven GRU / LSTM Transfer Learning Script
# ============================================================

import os
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# ===== 导入纯数据驱动模型 =====
from GRU_multi_step import PureGRUModel, StandardWeightedLoss

try:
    from LSTM_multi_step import PureLSTMModel
except ImportError:
    PureLSTMModel = None


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------

def infer_rnn_type_from_ckpt(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    for k in sd.keys():
        if "gru." in k:
            return "GRU"
        if "lstm." in k:
            return "LSTM"
    raise RuntimeError("Cannot infer RNN type from checkpoint")


def infer_input_size_from_ckpt(ckpt_path):
    sd = torch.load(ckpt_path, map_location="cpu")
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    for k in sd.keys():
        if "weight_ih_l0" in k:
            return sd[k].shape[1]
    raise RuntimeError("Cannot infer input_size from checkpoint")


def inverse_transform_y(preds_list, obs_list, scaler_y):
    """
    preds_list, obs_list: list of numpy arrays [B, pre_len, 1]
    returns: preds, obs in physical scale, shape [N*pre_len, 1]
    """
    preds_arr = np.vstack(preds_list)
    obs_arr   = np.vstack(obs_list)

    preds_2d = preds_arr.reshape(-1, 1)
    obs_2d   = obs_arr.reshape(-1, 1)

    preds_log = scaler_y.inverse_transform(preds_2d)
    obs_log   = scaler_y.inverse_transform(obs_2d)

    preds = np.expm1(preds_log)
    obs   = np.expm1(obs_log)

    return preds, obs



def build_aligned_x_with_obs(df, pretrained_model_path):
    """
    DW: 8 rainfall + OBS
    Source: 14 rainfall + OBS
    -> areal mean rainfall repeated to 14 + OBS
    """
    # infer source input structure
    src_input_size = infer_input_size_from_ckpt(pretrained_model_path)
    src_rain_dim = src_input_size - 1

    # extract raw data
    rain_raw = df.iloc[:, 1:-1].values.astype(float)   # [T, 8]
    obs_raw = df.iloc[:, -1].values.reshape(-1, 1)     # [T, 1]

    # areal precipitation
    rain_mean = np.mean(rain_raw, axis=1, keepdims=True)  # [T,1]

    # repeat rainfall
    rain_aligned = np.repeat(rain_mean, src_rain_dim, axis=1)  # [T,14]

    # append OBS_HIS
    X_raw = np.hstack([rain_aligned, obs_raw])  # [T,15]

    print(
        f"[DataAlign] DW rain={rain_raw.shape[1]} -> "
        f"areal mean -> repeat {src_rain_dim}, + OBS_HIS"
    )

    return rain_aligned, obs_raw, obs_raw



def make_sliding_window(x, y, window, pre_len):
    X, Y = [], []
    for i in range(len(x) - window - pre_len + 1):
        X.append(x[i:i + window])
        Y.append(y[i + window:i + window + pre_len])
    return np.array(X), np.array(Y)


def recover_Q_from_norm_log(y_norm, scaler_y):
    """
    y_norm: np.ndarray, shape [N, pre_len, 1] or [N, 1]
    returns: Q in physical scale
    """
    y_arr = np.asarray(y_norm)

    # reshape to 2D
    y_2d = y_arr.reshape(-1, 1)

    # inverse standard scaler -> log space
    y_log = scaler_y.inverse_transform(y_2d)

    # inverse log1p
    y_phys = np.expm1(y_log)

    # numerical safety
    y_phys = np.maximum(y_phys, 0.0)

    # reshape back
    return y_phys.reshape(y_arr.shape)


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_transfer(args):

    device = torch.device(args.device)

    # ===== 1. Load DW data =====
    df = pd.read_csv(args.dw_csv)

    # ===== 1.1 Build aligned inputs (14 rain + 1 OBS) =====
    rain_aligned, obs_raw, y_raw = build_aligned_x_with_obs(
        df,
        pretrained_model_path=args.pretrained_model
    )

    # ===== 2. Scaler（目标流域重新拟合） =====
    scaler_x = StandardScaler()
    scaler_y = StandardScaler()

    rain_n = scaler_x.fit_transform(rain_aligned)

    # 对流量取 log 再归一化（非常重要）
    obs_log = np.log1p(np.maximum(obs_raw, 0.0))
    obs_n = scaler_y.fit_transform(obs_log)

    # 2.3 拼接：输入特征 = [rain(14), OBS_HIS(1)]，且 OBS_HIS 与 y 在同一空间
    x = np.hstack([rain_n, obs_n])  # [T, 15]

    # 2.4 训练目标 y：同 obs_n（保证完全一致）
    y = obs_n

    # ===== 3. Sliding window =====
    X, Y = make_sliding_window(x, y, args.window, args.pre_len)

    n_total = len(X)
    n_train = int(n_total * args.train_ratio)
    n_valid = int(n_total * args.valid_ratio)

    X_train, Y_train = X[:n_train], Y[:n_train]
    X_valid, Y_valid = X[n_train:n_train + n_valid], Y[n_train:n_train + n_valid]
    X_test,  Y_test  = X[n_train + n_valid:], Y[n_train + n_valid:]

    print(f"[Split] train={len(X_train)} valid={len(X_valid)} test={len(X_test)}")

    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32),
                      torch.tensor(Y_train, dtype=torch.float32)),
        batch_size=args.batch,
        shuffle=True
    )
    valid_loader = DataLoader(
        TensorDataset(torch.tensor(X_valid, dtype=torch.float32),
                      torch.tensor(Y_valid, dtype=torch.float32)),
        batch_size=args.batch,
        shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32),
                      torch.tensor(Y_test, dtype=torch.float32)),
        batch_size=args.batch,
        shuffle=False
    )

    # ===== 4. Infer source model info =====
    rnn_type = infer_rnn_type_from_ckpt(args.pretrained_model)
    input_size = infer_input_size_from_ckpt(args.pretrained_model)

    print(f"[Info] Source model: {rnn_type}, input_size={input_size}")

    # ===== 5. Build model =====
    if rnn_type == "GRU":
        ModelClass = PureGRUModel
    elif rnn_type == "LSTM":
        if PureLSTMModel is None:
            raise RuntimeError("PureLSTMModel not found")
        ModelClass = PureLSTMModel
    else:
        raise RuntimeError("Unsupported RNN type")

    model = ModelClass(
        input_size=input_size,
        output_size=1,
        hidden_size=args.hidden,
        num_layers=args.layers,
        pre_len=args.pre_len,
        dropout=args.dropout
    ).to(device)

    # ===== 6. Load pretrained weights =====
    sd = torch.load(args.pretrained_model, map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    model.load_state_dict(sd, strict=True)

    print("[Load] pretrained data-driven model loaded")

    # ===== 7. Freeze strategy =====
    if args.freeze_backbone:
        for p in model.parameters():
            p.requires_grad = False

        if rnn_type == "GRU":
            for p in model.gru.parameters():
                p.requires_grad = True
        else:
            for p in model.lstm.parameters():
                p.requires_grad = True

        for p in model.fc_out.parameters():
            p.requires_grad = True

        print("[Transfer-DD] Freeze backbone, finetune RNN last layers + fc_out")
    else:
        print("[Transfer-DD] Finetune full model")

    # ===== 8. Loss & Optimizer =====
    criterion = StandardWeightedLoss(device=device, scaler_y=scaler_y)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # ===== 9. Training loop =====
    best_val = -1e9
    patience = 0

    for ep in range(1, args.epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            y_pred = model(xb)
            loss = criterion(y_pred, yb, xb)
            loss.backward()
            optimizer.step()

        # ---- Validation ----
        model.eval()
        preds, obs = [], []
        with torch.no_grad():
            for xb, yb in valid_loader:
                xb = xb.to(device)
                yp = model(xb)
                preds.append(yp.cpu().numpy())
                obs.append(yb.numpy())

        # ===== reshape to 2D for inverse_transform =====
        preds, obs = inverse_transform_y(preds, obs, scaler_y)

        nse = 1 - np.sum((preds - obs) ** 2) / np.sum((obs - obs.mean()) ** 2)

        print(f"Epoch {ep:03d} | Val NSE={nse:.3f}")

        if nse > best_val:
            best_val = nse
            patience = 0
            torch.save(model.state_dict(), args.out_model)
        else:
            patience += 1
            if patience >= args.patience:
                print("[EarlyStop]")
                break

    # ===== 10. Test =====
    model.load_state_dict(torch.load(args.out_model))
    model.eval()

    preds, obs = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yp = model(xb)
            preds.append(yp.cpu().numpy())
            obs.append(yb.numpy())

    # ===== 正确的时间序列构造：只取 lead = 1 =====
    preds_arr = np.vstack(preds)  # [N, pre_len, 1]
    obs_arr = np.vstack(obs)

    pred_lead1 = preds_arr[:, 0, 0]
    obs_lead1 = obs_arr[:, 0, 0]

    pred_lead1 = recover_Q_from_norm_log(pred_lead1, scaler_y)
    obs_lead1 = recover_Q_from_norm_log(obs_lead1, scaler_y)

    out = pd.DataFrame({
        "OBS": obs_lead1.ravel(),
        "SIM": pred_lead1.ravel()
    })
    out.to_csv(args.out_csv, index=False)

    # ===== Test NSE (use reconstructed lead-1 series) =====
    test_nse = 1.0 - np.sum((pred_lead1 - obs_lead1) ** 2) / np.sum(
        (obs_lead1 - obs_lead1.mean()) ** 2
    )
    print(f"[Test] NSE={test_nse:.3f}")

    print(f"[Output] saved to {args.out_csv}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dw_csv", type=str, default='.\\data\\Random_scenarios_DW.csv')
    p.add_argument("--pretrained_model", type=str, default="best_model.pth")
    p.add_argument("--window", type=int, default=72)
    p.add_argument("--pre_len", type=int, default=12)
    p.add_argument("--train_ratio", type=float, default=0.65)
    p.add_argument("--valid_ratio", type=float, default=0.2)
    p.add_argument("--hidden", type=int, default=16)
    p.add_argument("--layers", type=int, default=1)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_model", default="best_dd_transfer_model.pth")
    p.add_argument("--out_csv", default="dd_transfer_test_predictions.csv")

    args = p.parse_args()
    train_transfer(args)


if __name__ == "__main__":
    main()
