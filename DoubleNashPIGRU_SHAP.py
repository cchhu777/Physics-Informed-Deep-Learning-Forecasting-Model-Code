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
import torch.nn.functional as F
from torch.utils.data import Dataset
import seaborn as sns
import pickle


# Set the style for scientific research graphics
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
# ---------------------- 1. Physics Loss module (Coeff + Nash) ----------------------
class TimeSeriesDataset(Dataset):
    def __init__(self, sequences): self.sequences = sequences

    def __len__(self): return len(self.sequences)

    def __getitem__(self, index): return self.sequences[index]
class PhysicsInformedLoss(nn.Module):
    def __init__(self, area_km2, station_weights=[0.073, 0.066, 0.085, 0.135, 0.075, 0.073, 0.113, 0.04, 0.09, 0.057, 0.062, 0.071, 0.063, 0],
                 dt_hours=1.0, weight_mse=0.5, weight_phys=0.5, w_log=0.5, device='cuda', scaler_x=None, scaler_y=None):
        super(PhysicsInformedLoss, self).__init__()
        self.device = device
        self.mse = nn.MSELoss()
        self.area_km2 = area_km2
        # 单位换算: mm/h * unit_cov -> m^3/s
        self.unit_cov = (area_km2 * 1e6) / (1000.0 * 3600.0 * dt_hours)
        self.w_mse = weight_mse
        self.w_phys = weight_phys
        self.w_log = w_log

        # --- learnable Phys paras ---
        # 1. runoff coeff alpha: range (0, 1]
        # Initialization raw=0.0 -> sigmoid=0.5.
        self.alpha_raw = nn.Parameter(torch.tensor(0.5, device=device))

        # baseflow
        self.baseflow_raw = nn.Parameter(torch.tensor(-2.0, device=device))
        # self.baseflow_bias = nn.Parameter(torch.tensor(0.1, device=device))

        # 2. Nash 参数: n (形状), k (滞后)
        # 初始化 n=3, k=5
        self.Nash_n_raw = nn.Parameter(torch.tensor(6.0, device=device))  # 6
        self.Nash_k_raw = nn.Parameter(torch.tensor(8.0, device=device))  # 8
        self.Nash_n2_raw = nn.Parameter(torch.tensor(15.0, device=device))  # 15
        self.Nash_k2_raw = nn.Parameter(torch.tensor(20.0, device=device))  # 20
        # mix weight w = sigmoid(mix_raw) ∈ (0,1)，w越大越偏快流(uh1)
        self.mix_raw = nn.Parameter(torch.tensor(1.0, device=device))
        self.delay_raw = nn.Parameter(torch.tensor(15.0, device=device))  # lag time  # 15

        # Scaler buffers
        self.has_scaler = False
        if scaler_x is not None and scaler_y is not None:
            self.has_scaler = True
            x_mean = torch.tensor(scaler_x.mean_, dtype=torch.float32, device=device).view(1, 1, -1)
            x_scale = torch.tensor(scaler_x.scale_, dtype=torch.float32, device=device).view(1, 1, -1)
            self.register_buffer('x_mean', x_mean)
            self.register_buffer('x_scale', x_scale)
            y_mean = torch.tensor(scaler_y.mean_, dtype=torch.float32, device=device).view(1, 1, 1)
            y_scale = torch.tensor(scaler_y.scale_, dtype=torch.float32, device=device).view(1, 1, 1)
            self.register_buffer('y_mean', y_mean)
            self.register_buffer('y_scale', y_scale)

        if station_weights is not None:
            w = torch.tensor(station_weights, dtype=torch.float32, device=device)
            w = w / (torch.sum(w) + 1e-6)
            self.register_buffer('weights', w)
        else:
            self.weights = None

    def get_nash_uh(self, n, k, length=72):
        device = self.device
        t = torch.arange(length, dtype=torch.float32, device=device).view(1, 1, -1)

        n = torch.clamp(n, min=1.05, max=25.0)
        k = torch.clamp(k, min=3.0, max=40.0)

        log_k = torch.log(k)
        lgamma_n = torch.lgamma(n)
        t_safe = t + 0.1  # Prevent the t=0 problem
        log_t_div_k = torch.log(t_safe / k)

        # Gamma distribution formula calculation in the log domain
        log_uh = -log_k - lgamma_n + (n - 1) * log_t_div_k - (t_safe / k)

        # Prevent overflow by truncation
        log_uh = torch.clamp(log_uh, min=-30.0, max=10.0)
        uh = torch.exp(log_uh)

        # Normalization of the unit line to ensure water balance
        uh = uh / (torch.sum(uh) + 1e-8)
        return uh

    def get_physical_params(self):
        # alpha in [0.05, 1.0]
        alpha = torch.sigmoid(self.alpha_raw) * 0.95 + 0.05

        # Nash coeff constraints
        # n = torch.relu(self.Nash_n_raw) + 1.1
        # k = torch.relu(self.Nash_k_raw) + 0.5
        n = torch.clamp(torch.relu(self.Nash_n_raw) + 1.1, max=25.0)
        k = torch.clamp(torch.relu(self.Nash_k_raw) + 0.5, max=25.0)
        return alpha, n, k

    def get_physical_params_double(self):
        n2 = torch.clamp(torch.relu(self.Nash_n2_raw) + 1.1, max=25.0)
        k2 = torch.clamp(torch.relu(self.Nash_k2_raw) + 0.5, max=40.0)

        w = torch.sigmoid(self.mix_raw)
        return n2, k2, w

    def compute_simulated_feature(self, x_input):
        # 1. Inverse normalization of precip
        x_rain_norm = x_input[:, :, :-1]
        rain_phys_all = torch.relu(x_rain_norm * self.x_scale + self.x_mean)

        # 2. Average precip
        if self.weights is not None and rain_phys_all.shape[2] == self.weights.shape[0]:
            rain_phys_mean = torch.sum(rain_phys_all * self.weights, dim=2)
        else:
            rain_phys_mean = torch.mean(rain_phys_all, dim=2)

        # 3. gain the phys paras
        alpha, n, k = self.get_physical_params()

        # 4. runoff calculation: R = P * alpha
        runoff_seq = rain_phys_mean * alpha
        # confluence delay
        # τ >= 0
        tau = torch.clamp(F.softplus(self.delay_raw), 15.0, 24.0)
        shift = torch.round(tau).long()

        B, T = runoff_seq.shape

        # padding
        runoff_padded = F.pad(runoff_seq.unsqueeze(1), (shift, 0))  # [B,1,T+shift]
        runoff_seq = runoff_padded[:, :, :T].squeeze(1)  # [B,T]

        # 5. confluence calculation: Nash conv
        uh_len = min(72, rain_phys_mean.shape[1])
        # uh = self.get_nash_uh(n, k, length=uh_len)
        # --- Double Nash ---
        n2, k2, w = self.get_physical_params_double()

        uh1 = self.get_nash_uh(n, k, length=uh_len)  # Fast response kernel
        uh2 = self.get_nash_uh(n2, k2, length=uh_len)  # Slow retreat kernel

        # mix kernal
        uh = w * uh1 + (1.0 - w) * uh2

        # Re-normalize and ensure water balance
        uh = uh / (torch.sum(uh) + 1e-8)

        runoff_input = runoff_seq.unsqueeze(1)  # [B, 1, T]
        # padding = uh_len - 1
        # q_routed_raw = F.conv1d(runoff_input, uh, padding=padding)
        # q_routed_mm = q_routed_raw[:, :, :rain_phys_mean.shape[1]].squeeze(1)
        runoff_padded = F.pad(runoff_input, (uh_len - 1, 0))
        # baseflow = F.softplus(self.baseflow_raw)
        # baseflow = torch.clamp(baseflow, min=0.0, max=5.0)
        baseflow = 1.0 * torch.sigmoid(self.baseflow_raw)  # 限制基流不超过1
        # padding=0 to conv
        q_routed_raw = F.conv1d(runoff_padded, uh, padding=0)

        # At this point, the output length is exactly equal to the original input length, and it strictly corresponds to time t.
        q_routed_mm = q_routed_raw.squeeze(1)

        # convert to discharge m^3/s
        q_sim_phys = q_routed_mm * self.unit_cov
        q_sim_phys = q_sim_phys + baseflow

        # 6. normalized (Log -> Standard) for GRU
        q_sim_log = torch.log1p(q_sim_phys + 1e-6)
        q_sim_norm = (q_sim_log - self.y_mean.squeeze()) / self.y_scale.squeeze()

        return q_sim_norm.unsqueeze(2), q_sim_phys

    def forward(self, y_pred, y_true, x_input, mode='combined'):
        # calculate the simulated Q
        _, q_sim_phys = self.compute_simulated_feature(x_input)

        # get the obs Q
        x_obs_norm = x_input[:, :, -1]
        x_obs_log = x_obs_norm * self.y_scale + self.y_mean
        x_obs_phys = torch.relu(torch.expm1(x_obs_log))  # 真实的 m^3/s

        # Phys Loss: compare log(Q_sim) and log(Q_obs)
        # add 1e-6 to prevent log(0)
        q_sim_log_safe = torch.log1p(q_sim_phys + 1e-6)
        q_obs_log_safe = torch.log1p(x_obs_phys + 1e-6)

        # loss_phys_calibration = torch.mean((q_sim_log_safe - q_obs_log_safe) ** 2)
        loss_phys_calibration = self.mse(q_sim_log_safe, q_obs_log_safe)

        if mode == 'physics_only':
            return loss_phys_calibration

        # mix Loss
        if self.has_scaler:
            y_true_log = y_true * self.y_scale + self.y_mean
            y_true_phys = torch.relu(torch.expm1(y_true_log))
            # peak weighting
            beta = 0.8
            w_max = 2

            weights = 1.0 + beta * torch.log1p(y_true_phys)
            weights = torch.clamp(weights, 1.0, w_max)
            # weights = 1.0 + torch.log1p(y_true_phys)
            loss_mse = torch.mean(weights * (y_pred - y_true) ** 2)
        else:
            loss_mse = self.mse(y_pred, y_true)

        return self.w_mse * loss_mse + self.w_phys * loss_phys_calibration


# ---------------------- 2. Auxiliary function ----------------------
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------- 3. Data loader ----------------------
def create_dataloader(config, device):
    print(">>>>>>>>>>>>>>>>>>>>>>>>>>>> Load and process the data <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    df = pd.read_csv(config.data_path)
    if 'tm' in df.columns: df = df.drop(columns=['tm'])
    cols = [c for c in df.columns if c != config.target] + [config.target]
    df = df[cols]
    data_values = df.values

    n_total = len(data_values)
    train_size = int(0.7 * n_total)
    valid_size = int(0.2 * n_total)

    train_raw = data_values[:train_size]
    valid_raw = data_values[train_size:train_size + valid_size]
    test_raw = data_values[train_size + valid_size:]

    scaler_x = StandardScaler()
    train_x_norm = scaler_x.fit_transform(train_raw[:, :-1])
    valid_x_norm = scaler_x.transform(valid_raw[:, :-1])
    test_x_norm = scaler_x.transform(test_raw[:, :-1])

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

    train_tensor = torch.FloatTensor(train_data_norm)
    valid_tensor = torch.FloatTensor(valid_data_norm)
    test_tensor = torch.FloatTensor(test_data_norm)

    def create_inout_sequences(input_data, tw, pre_len):
        inout_seq = []
        L = len(input_data)
        for i in range(L - tw - pre_len + 1):
            train_seq = input_data[i: i + tw]
            train_label = input_data[i + tw: i + tw + pre_len, -1:]
            inout_seq.append((train_seq, train_label))
        return inout_seq

    train_seq = create_inout_sequences(train_tensor, config.window_size, config.pre_len)
    valid_seq = create_inout_sequences(valid_tensor, config.window_size, config.pre_len)
    test_seq = create_inout_sequences(test_tensor, config.window_size, config.pre_len)

    # shuffle=False
    train_loader = DataLoader(TimeSeriesDataset(train_seq), batch_size=config.batch_size, shuffle=True, drop_last=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    valid_loader = DataLoader(TimeSeriesDataset(valid_seq), batch_size=config.batch_size, shuffle=False, drop_last=True,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(TimeSeriesDataset(test_seq), batch_size=config.batch_size, shuffle=False, drop_last=False,
                             num_workers=4, pin_memory=True, persistent_workers=True)

    return train_loader, valid_loader, test_loader, scaler_x, scaler_y


# ---------------------- 4. GRU ----------------------
class PureGRUModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers, pre_len, dropout=0.1):
        super(PureGRUModel, self).__init__()
        self.pre_len = pre_len
        self.mask_prob = 0.0
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
                          dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(hidden_size, pre_len * output_size)

    def forward(self, x):
        if self.training and self.mask_prob > 0:
            batch_size = x.size(0)
            probs = torch.rand(batch_size, 1, 1, device=x.device)
            mask = (probs > self.mask_prob).float()
            x = torch.cat([x[:, :, :-1], x[:, :, -1:] * mask], dim=2)
        out_gru, _ = self.gru(x)
        h_last = self.dropout(out_gru[:, -1, :])
        out = self.fc_out(h_last)
        return out.view(-1, self.pre_len, 1)


# ---------------------- 5. main training process ----------------------
def train(model, args, device, train_loader, valid_loader, scaler_x, scaler_y):
    # Instantiate the physical model and immediately transfer it to the GPU
    criterion = PhysicsInformedLoss(
        area_km2=args.area,
        station_weights=None,
        weight_mse=1.0,
        weight_phys=0.5,  # 物理 Loss 权重
        w_log=0.4,
        device=device,
        scaler_x=scaler_x, scaler_y=scaler_y
    ).to(device)

    print(f"DEBUG: Model Device: {next(model.parameters()).device}")
    print(f"DEBUG: Physical Criterion Device: {criterion.alpha_raw.device}")

    optimizer_phys = torch.optim.Adam(criterion.parameters(), lr=0.001)  # leaning rate of phys paras

    optimizer_global = torch.optim.Adam([
        {'params': model.parameters(), 'lr': args.lr},
        {'params': criterion.parameters(), 'lr': 0.001}
    ])

    milestones = [int(args.epochs * 0.4), int(args.epochs * 0.6), int(args.epochs * 0.7), int(args.epochs * 0.9)]
    scheduler = MultiStepLR(optimizer_global, milestones=milestones, gamma=0.5)

    history = {'train': [], 'valid': []}

    # === Phase 1: Physical parameter preheating ===
    print(f"\n>>> [Phase 1] Physical parameter preheating ({args.warmup_epochs} Epochs) <<<")

    for i in range(args.warmup_epochs):
        epoch_loss = 0.0
        num_batches = 0
        for seq, labels in train_loader:
            seq = seq.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer_phys.zero_grad()
            loss = criterion(None, None, seq, mode='physics_only')
            loss.backward()
            optimizer_phys.step()
            # loss_list.append(loss.item())
            epoch_loss += loss.detach()
            num_batches += 1
        epoch_loss = (epoch_loss / num_batches).item()

        if (i + 1) % 10 == 0:
            with torch.no_grad():
                alpha, n, k = criterion.get_physical_params()
                n2, k2, w = criterion.get_physical_params_double()
                tau = torch.clamp(F.softplus(criterion.delay_raw), 0.0, 24.0)
            print(f"Warmup {i + 1}: Loss={epoch_loss:.4f} | "
                  f"Alpha={alpha:.3f}, n1={n:.2f}, k1={k:.2f}, n2={n2:.2f}, k2={k2:.2f}, w={w:.2f}")
            print(f"Delay τ = {tau.item():.2f} steps")

    # --- Visualization: The physical simulation effect after preheating ---
    print(">>> Draw the physical preheating effect diagram...")
    visualize_warmup(criterion, valid_loader, scaler_y)

    # === Phase 2: Joint Training ===
    print(f"\n>>> [Phase 2] Joint Training ({args.epochs} Epochs) <<<")

    best_valid_loss = float('inf')

    loop = tqdm(range(args.epochs))
    for i in loop:
        model.train()
        criterion.train()
        epoch_loss = 0.0
        num_batches = 0
        batch_train_losses = []
        for seq, labels in train_loader:
            seq = seq.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer_global.zero_grad()

            # Generate features using physical models
            q_sim_norm, _ = criterion.compute_simulated_feature(seq)

            # Input Replacement
            # seq_replaced = seq.clone()
            # seq_replaced[:, :, -1] = q_sim_norm.squeeze(2)
            seq_replaced = torch.cat([seq[:, :, :-1], q_sim_norm], dim=2)

            y_pred = model(seq_replaced)
            loss = criterion(y_pred, labels, seq, mode='combined')

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(criterion.parameters()),
                1.0
            )
            optimizer_global.step()
            # batch_train_losses.append(loss.item())
            epoch_loss += loss.detach()
            num_batches += 1

        scheduler.step()
        epoch_loss = (epoch_loss / num_batches).item()
        batch_train_losses.append(epoch_loss)
        history['train'].append(batch_train_losses)

        # Validation
        model.eval()
        criterion.eval()
        test_epoch_loss = 0.0
        test_num_batches = 0
        with torch.no_grad():
            for seq, labels in valid_loader:
                seq = seq.to(device)
                labels = labels.to(device)
                q_sim_norm, _ = criterion.compute_simulated_feature(seq)
                # seq_replaced = seq.clone()
                # seq_replaced[:, :, -1] = q_sim_norm.squeeze(2)
                seq_replaced = torch.cat([seq[:, :, :-1], q_sim_norm], dim=2)
                y_pred = model(seq_replaced)
                val_loss = criterion(y_pred, labels, seq, mode='combined')
                # batch_valid_losses.append(val_loss.item())
                test_epoch_loss += val_loss
                test_num_batches += 1

        epoch_valid_loss = (test_epoch_loss / test_num_batches).item()
        history['valid'].append(epoch_valid_loss)
        loop.set_postfix(T=batch_train_losses, V=epoch_valid_loss)

        if epoch_valid_loss < best_valid_loss:
            best_valid_loss = epoch_valid_loss
            torch.save(model.state_dict(), 'best_model.pth')
            torch.save(criterion.state_dict(), 'best_criterion.pth')
            with open("scaler_x.pkl", "wb") as f:
                pickle.dump(scaler_x, f)

            with open("scaler_y.pkl", "wb") as f:
                pickle.dump(scaler_y, f)

    # --- Visualization: Loss Curve ---
    plt.figure(figsize=(6, 4))
    plt.plot(history['train'], label='Train Loss')
    plt.plot(history['valid'], label='Valid Loss')
    plt.title("Training Progress")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('training_loss_curve.png')
    plt.show()

    # Reload the optimal physical parameters
    if os.path.exists('best_criterion.pth'):
        criterion.load_state_dict(torch.load('best_criterion.pth'))

    return criterion


def visualize_warmup(criterion, loader, scaler_y):
    print("\n" + "=" * 50)
    print("=" * 50 + "\n")

    criterion.eval()

    # print paras
    with torch.no_grad():
        alpha, n, k = criterion.get_physical_params()
        print(f"DEBUG: Params -> Alpha={alpha.item():.4f}, n={n.item():.4f}, k={k.item():.4f}")

    sim_list = []
    obs_list = []
    rain_list = []

    with torch.no_grad():
        for seq, _ in loader:
            seq = seq.to(criterion.alpha_raw.device, non_blocking=True)
            _, q_sim_phys = criterion.compute_simulated_feature(seq)

            x_obs_norm = seq[:, :, -1]
            x_obs_log = x_obs_norm * criterion.y_scale + criterion.y_mean
            x_obs_phys = torch.relu(torch.expm1(x_obs_log))

            x_rain_norm = seq[:, :, :-1]
            rain_phys_all = torch.relu(x_rain_norm * criterion.x_scale + criterion.x_mean)
            rain_phys_mean = torch.mean(rain_phys_all, dim=2)

            s_batch = q_sim_phys[:, -1]
            o_batch = x_obs_phys[:, -1]
            r_batch = rain_phys_mean[:, -1]

            # 2. convert to numpy
            sim_list.append(s_batch.cpu().numpy().reshape(-1))
            obs_list.append(o_batch.cpu().numpy().reshape(-1))
            rain_list.append(r_batch.cpu().numpy().reshape(-1))

    # concat
    sim = np.concatenate(sim_list)
    obs = np.concatenate(obs_list)
    rain = np.concatenate(rain_list)

    # --- [Dimension Diagnosis] Printing the shape of the final data ---
    print(f"DEBUG: Final Data Shapes -> Obs: {obs.shape}, Sim: {sim.shape}")
    if len(obs.shape) > 1:
        print("Anomaly detection: The data is still multi-dimensional! It is being forced to flatten....")
        obs = obs.flatten()
        sim = sim.flatten()
        rain = rain.flatten()

    # Plot
    limit = min(600, len(obs))

    plt.close('all')
    fig, ax1 = plt.subplots(figsize=(6, 4))

    ax2 = ax1.twinx()
    ax2.bar(range(limit), rain[:limit], color='gray', alpha=0.3, label='Rainfall')
    ax2.set_ylabel('Rainfall (mm)')
    ax2.set_ylim(0, max(rain[:limit].max() * 3, 10))
    ax2.invert_yaxis()

    ax1.plot(obs[:limit], label='Observed Flow', color='black', linewidth=1.5)
    ax1.plot(sim[:limit], label=f'Physical Sim (k={k.item():.1f})', color='blue', linestyle='--', linewidth=1.5)

    ax1.set_xlabel('Time Steps (Continuous)')
    ax1.set_ylabel('Flow (m3/s)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_title(f'Physical Check: Alpha={alpha.item():.3f}, k={k.item():.2f}')

    plt.savefig('warmup_check_DEBUG.png')
    print(">>> Plot finished, check warmup_check_DEBUG.png")


# ---------------------- 6. Test  ----------------------
def test(model, test_loader, scaler_y, criterion=None):
    if not os.path.exists('best_model.pth'): return
    model.load_state_dict(torch.load('best_model.pth'))
    if os.path.exists('best_criterion.pth') and criterion is not None:
        criterion.load_state_dict(torch.load('best_criterion.pth'))

    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for seq, label in test_loader:
            seq = seq.to(next(model.parameters()).device, non_blocking=True)
            label = label.to(next(model.parameters()).device, non_blocking=True)
            q_sim_norm, _ = criterion.compute_simulated_feature(seq)
            # seq_replaced = seq.clone()
            # seq_replaced[:, :, -1] = q_sim_norm.squeeze(2)
            seq_replaced = torch.cat([seq[:, :, :-1], q_sim_norm], dim=2)

            pred = model(seq_replaced)  # [B, T, 1]

            # renormalization
            B, T, C = pred.shape
            pred_inv = scaler_y.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).reshape(B, T, C)
            label_inv = scaler_y.inverse_transform(label.cpu().numpy().reshape(-1, 1)).reshape(B, T, C)

            preds.append(np.expm1(pred_inv))
            trues.append(np.expm1(label_inv))

    preds_arr = np.concatenate(preds, axis=0)  # [N, T, 1]
    trues_arr = np.concatenate(trues, axis=0)

    # get the last time step
    sim = preds_arr[:, -1, 0]
    obs = trues_arr[:, -1, 0]

    nse = 1 - np.sum((sim - obs) ** 2) / (np.sum((obs - np.mean(obs)) ** 2) + 1e-6)
    print(f"Test NSE (Final Step): {nse:.4f}")

    plt.figure(figsize=(12, 6))
    plt.plot(obs[:300], label='Observed', color='black', alpha=0.7)
    plt.plot(sim[:300], label=f'Predicted (NSE={nse:.3f})', color='red', linestyle='--')
    plt.title("Test Set Prediction")
    plt.legend()
    plt.grid(True)
    plt.savefig('test_result.png')
    plt.show()

    df_res = pd.DataFrame({'Obs': obs, 'Pred': sim})
    df_res.to_csv('predict_result.csv', index=False)
    print("Saved")



# ---------------------- 7. SHAP ----------------------
def run_shap_analysis(model, criterion, loader, args, scaler_y, device,
                      max_background=200, max_explain=100, seed=42):
    try:
        import shap
    except Exception as e:
        print(f"Error: {e}")
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
        print(f"Error: {e}")

        feature_names = None

    model.eval()
    criterion.eval()

    # --------- 1) Extract background samples and explanation samples from the loader ---------
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

    # fill feature_names
    if feature_names is None:
        Fdim = background.shape[-1]
        feature_names = [f"X{i}" for i in range(Fdim - 1)] + [args.target + "_HIS"]

    # --------- 2) Define the "end-to-end" prediction function---------
    class End2EndWrapper(nn.Module):
        def __init__(self, model, criterion):
            super().__init__()
            self.model = model
            self.criterion = criterion

        def forward(self, x_input):
            # Generate features using physical models
            q_sim_norm, _ = self.criterion.compute_simulated_feature(x_input)
            # Input Replacement
            seq_replaced = torch.cat([x_input[:, :, :-1], q_sim_norm], dim=2)
            y_pred = self.model(seq_replaced)  # [B, pre_len, 1]
            out = y_pred[:, -1, 0]
            return out.unsqueeze(1)

    wrapped = End2EndWrapper(model, criterion).to(device)
    # wrapped.eval()
    model_was_training = model.training
    crit_was_training = criterion.training

    # --------- 3) Calculate SHAP value---------
    shap_values = None
    used_method = None

    # DeepExplainer / GradientExplainer
    try:
        # explainer = shap.DeepExplainer(wrapped, background)
        # sv = explainer.shap_values(explain_x)
        # shap_values = sv[0] if isinstance(sv, list) else sv
        # used_method = "DeepExplainer"
        model.train()
        criterion.train()
        wrapped.train()

        for m in wrapped.modules():
            if isinstance(m, torch.nn.Dropout):
                m.eval()

        background_t = background.detach()
        explain_t = explain_x.detach()

        with torch.backends.cudnn.flags(enabled=False):
            explainer = shap.DeepExplainer(wrapped, background_t)
            sv = explainer.shap_values(explain_t, check_additivity=False)

        shap_values = sv[0] if isinstance(sv, list) else sv
        used_method = "DeepExplainer"

    except Exception as e_deep:
        print(f"DeepExplainer is unavailable. Try KernelExplainer (it will be slower). Reason: {e_deep}")

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
        # Restore to original state
        model.train(model_was_training)
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

    # --------- 4) Importance aggregation---------
    # per_feature_importance: [N,F]
    per_feature_importance = np.abs(shap_values).sum(axis=1)
    per_feature_value = x_values.mean(axis=1)  # [N,F] 用时间均值表示“特征取值”
    mean_abs_importance = per_feature_importance.mean(axis=0)  # [F]

    # --------- 5) Plot 1：Global Feature Importance---------
    order = np.argsort(-mean_abs_importance)
    topk = min(20, len(order))
    top_idx = order[:topk]
    plt.figure(figsize=(6, 4))
    plt.bar(range(topk), mean_abs_importance[top_idx])
    plt.xticks(range(topk), [feature_names[i] for i in top_idx], rotation=45, ha='right')
    plt.ylabel("mean(|SHAP|) over time")
    plt.title("Global Feature Importance")
    plt.tight_layout()
    # plt.savefig(".shap_global_importance(PIGRU-0.7TS).png", dpi=300)
    plt.show()

    # --------- 6) Plot 2：SHAP beeswarm ---------
    try:
        import shap
        shap.summary_plot(
            per_feature_importance,
            features=per_feature_value,
            feature_names=feature_names,
            show=False
        )
        plt.title("SHAP Summary (Aggregated over time)")
        plt.tight_layout()
        # plt.savefig(".\\shap_summary_aggregated.png", dpi=300)
        plt.show()
    except Exception as e:
        print(f"[SHAP] summary_plot failure! Reason: {e}")

    # --------- 7) Plot 3：Temporal Importance ---------
    sample_id = 0
    time_importance = np.abs(shap_values[sample_id]).sum(axis=1)  # [T]
    plt.figure(figsize=(8, 4))
    plt.plot(time_importance)
    plt.xlabel("Time step (within window)")
    plt.ylabel("sum(|SHAP|) over features")
    plt.title("Temporal Importance (one sample)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig("shap_temporal_importance_one_sample.png", dpi=300)
    plt.show()

    print("[SHAP] Saved："
          "shap_global_importance.png, shap_summary_aggregated.png, shap_temporal_importance_one_sample.png")


# ---------------------- 7. 主程序 ----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # HyperParams
    parser.add_argument('-data_path', type=str, default='.\\data\\Random_scenarios.csv')
    parser.add_argument('-target', type=str, default='OBS')
    parser.add_argument('-area', type=float, default=4623.0)
    parser.add_argument('-window_size', type=int, default=72)
    parser.add_argument('-pre_len', type=int, default=12)
    parser.add_argument('-warmup_epochs', type=int, default=200)
    parser.add_argument('-epochs', type=int, default=200)
    parser.add_argument('-batch_size', type=int, default=64)
    parser.add_argument('-lr', type=float, default=0.001)
    parser.add_argument('-hidden_size', type=int, default=16)
    parser.add_argument('-layers', type=int, default=1)

    args = parser.parse_args()

    # Check GPU
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(">>> Success: CUDA is available. Using GPU.")
    else:
        device = torch.device("cpu")
        print(">>> Warning: CUDA not found. Using CPU.")

    seed_everything(42)

    if not os.path.exists(args.data_path):
        print(f"File not found: {args.data_path}")
        exit()

    # Loading
    train_loader, valid_loader, test_loader, scaler_x, scaler_y = create_dataloader(args, device)

    sample_seq, _ = next(iter(train_loader))
    input_size = sample_seq.shape[2]

    # initialized
    model = PureGRUModel(input_size=input_size, output_size=1,
                         hidden_size=args.hidden_size, num_layers=args.layers,
                         pre_len=args.pre_len).to(device)

    # Train
    trained_criterion = train(model, args, device, train_loader, valid_loader, scaler_x, scaler_y)

    # test
    test(model, test_loader, scaler_y, trained_criterion)
    # --- SHAP ---
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth', map_location=device))
    if os.path.exists('best_criterion.pth'):
        trained_criterion.load_state_dict(torch.load('best_criterion.pth', map_location=device))

    # Use test_loader to explain
    run_shap_analysis(model, trained_criterion, test_loader, args, scaler_y, device,
                      max_background=200, max_explain=100, seed=42)
