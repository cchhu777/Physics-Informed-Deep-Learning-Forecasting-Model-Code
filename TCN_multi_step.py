import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import time
from torch.optim.lr_scheduler import MultiStepLR
import random
import os
from sklearn.preprocessing import MinMaxScaler
from permetrics.regression import RegressionMetric


import torch
import torch.nn as nn
import seaborn as sns


plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 9,
    "figure.titlesize": 12,
    "font.family": "Arial",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})
sns.set_style("ticks")


class StandardWeightedLoss(nn.Module):
    def __init__(self, device='cpu', scaler_y=None):
        super(StandardWeightedLoss, self).__init__()
        self.device = device
        self.mse = nn.MSELoss()

        self.has_scaler = False
        if scaler_y is not None:
            self.has_scaler = True
            y_mean = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=device).view(1, 1, 1)
            y_scale = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=device).view(1, 1, 1)
            self.register_buffer('y_mean', y_mean)
            self.register_buffer('y_scale', y_scale)

    def forward(self, y_pred, y_true, x_input):

        if self.has_scaler:

            y_true_log = y_true * self.y_scale + self.y_mean
            y_true_phys = torch.expm1(y_true_log)  # exp(x) - 1
            y_true_phys = torch.relu(y_true_phys)

            weights = 1.0 + torch.log1p(y_true_phys)

            loss = torch.mean(weights * (y_pred - y_true) ** 2)
            return loss
        else:
            return self.mse(y_pred, y_true)


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True


def create_dataloader(config, device):

    df = pd.read_csv(config.data_path)

    if 'tm' in df.columns:
        df = df.drop(columns=['tm'])

    cols = [c for c in df.columns if c != config.target] + [config.target]
    df = df[cols]

    print(f"Feature: {cols[:5]} ...")
    print(f"Obj: {cols[-1]}")

    data_values = df.values  # [Total_Len, Features]

    n_total = len(data_values)
    train_size = int(0.7 * n_total)
    valid_size = int(0.2 * n_total)

    train_raw = data_values[:train_size]
    valid_raw = data_values[train_size:train_size + valid_size]
    test_raw = data_values[train_size + valid_size:]

    scaler_x = StandardScaler()

    train_x = train_raw[:, :-1]
    valid_x = valid_raw[:, :-1]
    test_x = test_raw[:, :-1]

    # Fit on Train, Transform on All
    train_x_norm = scaler_x.fit_transform(train_x)
    valid_x_norm = scaler_x.transform(valid_x)
    test_x_norm = scaler_x.transform(test_x)

    train_y_log = np.log1p(train_raw[:, -1:])
    valid_y_log = np.log1p(valid_raw[:, -1:])
    test_y_log = np.log1p(test_raw[:, -1:])

    scaler_y = StandardScaler()

    train_y_norm = scaler_y.fit_transform(train_y_log)
    valid_y_norm = scaler_y.transform(valid_y_log)
    test_y_norm = scaler_y.transform(test_y_log)

    train_data_norm = np.hstack([train_x_norm, train_y_norm])
    valid_data_norm = np.hstack([valid_x_norm, valid_y_norm])
    test_data_norm = np.hstack([test_x_norm, test_y_norm])

    train_tensor = torch.FloatTensor(train_data_norm).to(device)
    valid_tensor = torch.FloatTensor(valid_data_norm).to(device)
    test_tensor = torch.FloatTensor(test_data_norm).to(device)

    def create_inout_sequences(input_data, tw, pre_len):
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw - pre_len + 1):
            # Input: [t, t+tw]
            train_seq = input_data[i: i + tw]
            # Label: [t+tw, t+tw+pre_len]
            train_label = input_data[i + tw: i + tw + pre_len, -1:]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    train_seq = create_inout_sequences(train_tensor, config.window_size, config.pre_len)
    valid_seq = create_inout_sequences(valid_tensor, config.window_size, config.pre_len)
    test_seq = create_inout_sequences(test_tensor, config.window_size, config.pre_len)

    # DataLoader
    train_loader = DataLoader(TimeSeriesDataset(train_seq), batch_size=config.batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(TimeSeriesDataset(valid_seq), batch_size=config.batch_size, shuffle=False, drop_last=True)
    test_loader = DataLoader(TimeSeriesDataset(test_seq), batch_size=config.batch_size, shuffle=False, drop_last=False)

    return train_loader, test_loader, valid_loader, scaler_x, scaler_y


class TimeSeriesDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        # x: [B,C,T]
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]


class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # causal padding

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        # x: [B,C,T]
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=3, dropout=0.2):
        """
        num_channels: list
        """
        super().__init__()
        layers = []
        for i in range(len(num_channels)):
            in_ch = num_inputs if i == 0 else num_channels[i - 1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            layers.append(TemporalBlock(in_ch, out_ch, kernel_size, dilation, dropout))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class PureTCNModel(nn.Module):

    def __init__(self, input_size, output_size, hidden_size, num_layers, pre_len, dropout=0.2):
        super(PureTCNModel, self).__init__()
        self.pre_len = pre_len
        self.output_size = output_size
        self.input_size = input_size

        # Anti-Lazy mask
        self.mask_prob = 0.9

        num_channels = [hidden_size] * max(1, num_layers)

        self.tcn = TemporalConvNet(
            num_inputs=input_size,
            num_channels=num_channels,
            kernel_size=3,
            dropout=dropout
        )

        self.dropout = nn.Dropout(dropout)

        self.fc_out = nn.Linear(num_channels[-1], pre_len * output_size)

    def forward(self, x):
        # x: [B, Window, Feat]

        # =================== Anti-Lazy Mechanism ===================
        if self.training:
            batch_size = x.size(0)
            device = x.device
            probs = torch.rand(batch_size, 1, 1, device=device)
            mask = (probs > self.mask_prob).float()

            x_rain = x[:, :, :-1]
            x_flow = x[:, :, -1:]
            x_flow_masked = x_flow * mask
            x = torch.cat([x_rain, x_flow_masked], dim=2)

        x_tcn = x.transpose(1, 2)  # [B, Feat, Window]
        out_tcn = self.tcn(x_tcn)  # [B, hidden, Window]

        h_last = out_tcn[:, :, -1]  # [B, hidden]
        h_last = self.dropout(h_last)

        out = self.fc_out(h_last)  # [B, pre_len*output]
        return out.view(-1, self.pre_len, self.output_size)


def train(model, args, device, train_loader, scaler_x, scaler_y):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    milestones = [
        int(args.epochs * 0.5),
        int(args.epochs * 0.75),
        int(args.epochs * 0.9)
    ]
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)

    criterion = StandardWeightedLoss(device=device, scaler_y=scaler_y)

    model.train()
    loss_history = []

    print("Start training...")
    for i in tqdm(range(args.epochs)):
        batch_losses = []
        for seq, labels in train_loader:
            optimizer.zero_grad()
            y_pred = model(seq)
            loss = criterion(y_pred, labels, seq)
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        scheduler.step()
        epoch_loss = np.mean(batch_losses)
        loss_history.append(epoch_loss)

    torch.save(model.state_dict(), 'best_model.pth')

    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.show()


def test(model, args, test_loader, scaler_y):
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for seq, label in test_loader:
            pred = model(seq)  # [Batch, pre_len, 1] (这是 Norm(Log(Q)) )

            pred_np = pred.cpu().numpy()
            label_np = label.cpu().numpy()

            # 1. Inverse Standard Scaler
            N, T, D = pred_np.shape

            pred_flat = pred_np.reshape(-1, 1)
            label_flat = label_np.reshape(-1, 1)

            pred_inv_log = scaler_y.inverse_transform(pred_flat)
            label_inv_log = scaler_y.inverse_transform(label_flat)

            # 2. Inverse Log (Exp)
            # y = exp(y_log) - 1
            pred_real = np.expm1(pred_inv_log)
            label_real = np.expm1(label_inv_log)

            pred_real = np.maximum(pred_real, 0.0)
            label_real = np.maximum(label_real, 0.0)

            # Reshape 回 [N, T, D] 并存入列表
            preds.append(pred_real.reshape(N, T, 1))
            trues.append(label_real.reshape(N, T, 1))

    # Concat
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(trues[:200, 0, 0], label='Observed')
    plt.plot(preds[:200, 0, 0], label='Predicted', linestyle='--')
    plt.legend()
    plt.title("Prediction Visualization")
    plt.show()

    df_res = pd.DataFrame({
        'Obs_t1': trues[:, 0, 0],
        'Pred_t1': preds[:, 0, 0]
    })
    df_res.to_csv('TCN_predict_result.csv', index=False)
    print("Results saved in TCN_predict_result.csv")


def run_shap_analysis(model, criterion, loader, args, scaler_y, device,
                      max_background=200, max_explain=100, seed=42):

    try:
        import shap
    except Exception as e:
        print(f"[SHAP] ERROR: {e}")
        return

    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        df = pd.read_csv(args.data_path)
        if 'tm' in df.columns:
            df = df.drop(columns=['tm'])
        feature_names = [c for c in df.columns if c != args.target]

        feature_names = feature_names + [args.target + "_HIS"]
    except Exception as e:
        print(f"[SHAP] ERROR: {e}")
        feature_names = None

    model.eval()
    if criterion is not None:
        criterion.eval()

    background_list = []
    explain_list = []
    for seq, label in loader:
        background_list.append(seq)
        if sum(x.shape[0] for x in background_list) >= max_background:
            break
    for seq, label in loader:
        explain_list.append(seq)
        if sum(x.shape[0] for x in explain_list) >= max_explain:
            break

    background = torch.cat(background_list, dim=0)[:max_background].to(device)
    explain_x = torch.cat(explain_list, dim=0)[:max_explain].to(device)

    if feature_names is None:
        Fdim = background.shape[-1]
        feature_names = [f"X{i}" for i in range(Fdim - 1)] + [args.target + "_HIS"]

    class End2EndWrapper(nn.Module):
        def __init__(self, model, criterion=None):
            super().__init__()
            self.model = model
            self.criterion = criterion

        def forward(self, x_input):
            if self.criterion is not None:
                q_sim_norm, _ = self.criterion.compute_simulated_feature(x_input)
                x_input = torch.cat([x_input[:, :, :-1], q_sim_norm], dim=2)

            y_pred = self.model(x_input)  # [B, pre_len, 1]
            out = y_pred[:, -1, 0]
            return out.unsqueeze(1)

    wrapped = End2EndWrapper(model, criterion).to(device)
    # wrapped.eval()
    model_was_training = model.training
    crit_was_training = criterion.training if criterion is not None else None

    shap_values = None
    used_method = None

    # DeepExplainer / GradientExplainer
    try:
        model.train()
        if criterion is not None:
            criterion.train()
        wrapped.train()

        for m in wrapped.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()

        background_t = background.detach()
        explain_t = explain_x.detach()

        with torch.backends.cudnn.flags(enabled=True):
            explainer = shap.DeepExplainer(wrapped, background_t)
            sv = explainer.shap_values(explain_t, check_additivity=False)

        shap_values = sv[0] if isinstance(sv, list) else sv
        used_method = "DeepExplainer"

    except Exception as e_deep:
        print(f"[SHAP] DeepExplainer is unavailable. Try KernelExplainer (it will be slower). Reason: {e_deep}")

        # --- KernelExplainer ---
        background_np = background.detach().cpu().numpy()  # (Nb, T, F)
        explain_np = explain_x.detach().cpu().numpy()  # (Ne, T, F)

        T = background_np.shape[1]
        Fdim = background_np.shape[2]

        background_2d = background_np.reshape(background_np.shape[0], -1)  # (Nb, T*F)
        explain_2d = explain_np.reshape(explain_np.shape[0], -1)  # (Ne, T*F)

        def predict_fn(x2d):
            # x2d: (N, T*F) -> reshape back to (N, T, F)
            x3d = x2d.reshape(-1, T, Fdim)
            x_t = torch.from_numpy(x3d).to(device).float()
            with torch.no_grad():
                out = wrapped(x_t)
            return out.detach().cpu().numpy()  # (N,1)

        explainer = shap.KernelExplainer(predict_fn, background_2d)
        shap_values = explainer.shap_values(explain_2d, nsamples=200)

        shap_values = shap_values[0] if isinstance(shap_values, list) else shap_values  # (Ne, T*F)
        shap_values = np.array(shap_values).reshape(-1, T, Fdim)

        used_method = "KernelExplainer"

    finally:
        model.train(model_was_training)
        if criterion is not None:
            criterion.train(crit_was_training)

    print(f"[SHAP] Method: {used_method} | shap_values shape={np.array(shap_values).shape}")

    # shap_values expect shape=[N,T,F]
    shap_values = np.array(shap_values)
    x_values = explain_x.detach().cpu().numpy()

    if shap_values.ndim == 4 and shap_values.shape[-1] == 1:
        shap_values = shap_values.squeeze(-1)  # -> (N,T,F)

    elif shap_values.ndim == 2:
        shap_values = shap_values.reshape(-1, T, Fdim)

    if shap_values.ndim != 3:
        print(f"[SHAP] Expect shap_values is 3D [N,T,F], but obtained {shap_values.shape}")
        return

    # per_feature_importance: [N,F]
    per_feature_importance = np.abs(shap_values).sum(axis=1)
    per_feature_value = x_values.mean(axis=1)
    mean_abs_importance = per_feature_importance.mean(axis=0)  # [F]

    order = np.argsort(-mean_abs_importance)
    topk = min(20, len(order))
    top_idx = order[:topk]
    plt.figure(figsize=(6, 4))
    plt.bar(range(topk), mean_abs_importance[top_idx])
    plt.xticks(range(topk), [feature_names[i] for i in top_idx], rotation=45, ha='right')
    plt.ylabel("mean(|SHAP|) over time")
    plt.title("Global Feature Importance (TCN)")
    plt.tight_layout()
    # plt.savefig(".\\TS0.7_global_importance.png", dpi=300)
    plt.show()

    try:
        import shap
        shap.summary_plot(
            per_feature_importance,
            features=per_feature_value,
            feature_names=feature_names,
            show=False
        )
        plt.title("SHAP Summary (TCN)")
        plt.tight_layout()
        # plt.savefig(".\\TS0.7_summary.png", dpi=300)
        plt.show()
    except Exception as e:
        print(f"[SHAP] ERROR: {e}")

    sample_id = 0
    time_importance = np.abs(shap_values[sample_id]).sum(axis=1)  # [T]
    plt.figure(figsize=(8, 4))
    plt.plot(time_importance)
    plt.xlabel("Time step (within window)")
    plt.ylabel("sum(|SHAP|) over features")
    plt.title("Temporal Importance (TCN)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("shap_temporal_importance_one_sample(TCN).png", dpi=300)
    plt.show()

    print("[SHAP] Saved："
          "shap_global_importance.png, shap_summary_aggregated.png, shap_temporal_importance_one_sample.png")


if __name__ == '__main__':
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Current Device: {torch.cuda.get_device_name(0)}")
    else:
        print("Training on CPU")

    parser = argparse.ArgumentParser()
    parser.add_argument('-data_path', type=str, default='.\\data\\Random_scenarios.csv')
    parser.add_argument('-target', type=str, default='OBS')
    parser.add_argument('-window_size', type=int, default=72, help="Historical length")
    parser.add_argument('-pre_len', type=int, default=12, help="lead time")
    parser.add_argument('-epochs', type=int, default=300)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-hidden_size', type=int, default=16)
    parser.add_argument('-device', type=int, default=0)
    parser.add_argument('-area', type=float, default=4623.0, help="Area (km3)")
    parser.add_argument('-input_size', type=int, default=15, help='Dimension of input')
    parser.add_argument('-output_size', type=int, default=1, help='Output dimension')

    parser.add_argument('-drop_out', type=float, default=0.2, help="Dropout rate")
    parser.add_argument('-layer_num', type=int, default=6, help="number of TCN blocks")

    parser.add_argument('-use_gpu', type=bool, default=True)
    parser.add_argument('-train', type=bool, default=True)

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")

    train_loader, test_loader, valid_loader, scaler_x, scaler_y = create_dataloader(args, device)

    sample_seq, _ = next(iter(train_loader))
    input_size = sample_seq.shape[2]
    print(f"Input dimension: {input_size}")

    model = PureTCNModel(input_size, args.output_size,
                     hidden_size=args.hidden_size,
                     num_layers=args.layer_num,
                     pre_len=args.pre_len,
                     dropout=args.drop_out).to(device)

    train(model, args, device, train_loader, scaler_x, scaler_y)
    test(model, args, test_loader, scaler_y)
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    model.eval()

    run_shap_analysis(model, None, test_loader, args, scaler_y, device,
                      max_background=200, max_explain=100, seed=42)




