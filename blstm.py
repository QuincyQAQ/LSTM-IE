import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import random

def set_seed(seed=23):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 多卡时使用
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(212)

# ==== 1. Logistic 映射生成 ====
r = 3.99
N = 1000
train_len = 500
x = [0.1]

for _ in range(N - 1):
    x_next = r * x[-1] * (1 - x[-1])
    x.append(x_next)

x_series = np.array(x)

# ==== 2. 构造 LSTM 训练数据 ====
seq_len = 10

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def prepare_data(series):
    X, Y = [], []
    for i in range(train_len - seq_len):
        X.append(series[i:i + seq_len])
        Y.append(series[i + seq_len])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1).to(device)
    return X, Y

Xx, Yx = prepare_data(x_series)

# ==== 3. 定义 LSTM 模型 ====
class BiLSTMForecast(nn.Module):
    def __init__(self, hidden_size):
        super(BiLSTMForecast, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=hidden_size,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        output = self.fc(last_out)
        return output

# ==== 4. 模型训练函数 ====
def train_model(X_train, Y_train, model_name, hidden_size=64, epochs=500, lr=1e-2):
    model = BiLSTMForecast(hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-4)

    loss_history = []
    lr_history = []

    for ep in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        current_lr = optimizer.param_groups[0]['lr']
        lr_history.append(current_lr)

        if (ep + 1) % 50 == 0:
            print(f'{model_name} - Epoch [{ep + 1}/{epochs}], Loss: {loss.item():.6f}, LR: {current_lr:.2e}')

    df = pd.DataFrame({
        'epoch': range(1, epochs + 1),
        'loss': loss_history,
        'learning_rate': lr_history
    })
    df.to_csv(f'{model_name}_training_history.csv', index=False)

    return model

model_x = train_model(Xx, Yx, 'logistic_model')

# ==== 5. LSTM 预测后续序列 ====
def predict_future(model, series, pred_len):
    model.eval()
    window = list(series[train_len - seq_len:train_len])
    preds = []
    for _ in range(pred_len):
        inp = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            pred = model(inp).item()
        preds.append(pred)
        window.pop(0)
        window.append(pred)
    return np.concatenate([series[:train_len], preds])

pred_len = N - train_len  # = 2000

x_new = predict_future(model_x, x_series, pred_len)


def zero_one_test_for_chaos(x, c=0.7, normalize=True):
    """
    0-1 test for chaos based on projection method.
    Parameters:
        x : 1D np.ndarray
            Input time series.
        c : float
            Random constant in (0, π) for projection.
        normalize : bool
            Whether to normalize the input series.
    Returns:
        K : float
            The 0-1 chaos test statistic.
    """
    if normalize:
        x = (x - np.mean(x)) / np.std(x)

    n = len(x)
    t = np.arange(1, n + 1)
    cs = np.cumsum(x * np.cos(c * t))
    ss = np.cumsum(x * np.sin(c * t))

    M = cs**2 + ss**2
    D = np.var(M)

    K = np.corrcoef(t, M)[0, 1]
    return K, M


# ==== 6. 绘图 ====
fig, ax = plt.subplots(figsize=(13, 4))
ax.plot(x_series, color='blue', label='$x_n$')
ax.plot(x_new, color='red', linestyle='--', label="$x_n\\prime$")
ax.axvline(train_len, color='gray', linestyle=':', linewidth=1)
ax.set_xlim(0, len(x_series))
ax.set_xlabel('$n$')
ax.set_ylabel('$x$')
ax.legend(loc='upper right')
ax.text(0.05, 0.9, '(a)', transform=ax.transAxes, fontsize=12)
plt.tight_layout()
plt.show()
# 构建 DataFrame，列出索引、真实序列、LSTM 混合预测序列
df_seq = pd.DataFrame({
    'index': np.arange(len(x_series)),
    'x_true': x_series,
    'x_mixed': x_new
})

# 保存为 CSV 文件
df_seq.to_csv('logistic_sequence_vs_lstm_prediction.csv', index=False)

print("✅ 序列已保存至 logistic_sequence_vs_lstm_prediction.csv")


# 运行 0-1 测试
K_true, M_true = zero_one_test_for_chaos(x_series)
K_pred, M_pred = zero_one_test_for_chaos(x_new)

print(f"✔️ Logistic 原始序列的 0-1 chaos K 值：{K_true:.4f}")
print(f"✔️ LSTM 预测混合序列的 0-1 chaos K 值：{K_pred:.4f}")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(M_true, label=f"K = {K_true:.3f}")
plt.title("M(n) - True Logistic Sequence")
plt.xlabel("n")
plt.ylabel("M(n)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(M_pred, label=f"K = {K_pred:.3f}", color='red')
plt.title("M(n) - LSTM Mixed Prediction")
plt.xlabel("n")
plt.ylabel("M(n)")
plt.legend()

plt.tight_layout()
plt.show()



# ==== 7. 生成“全预测”序列 ====
def predict_fully(model, initial_seq, total_len):
    model.eval()
    window = list(initial_seq[-seq_len:])
    preds = []
    for _ in range(total_len):
        inp = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        with torch.no_grad():
            pred = model(inp).item()
        preds.append(pred)
        window.pop(0)
        window.append(pred)
    return np.array(preds)

x_fully_pred = predict_fully(model_x, x_series[:train_len], N)

# ==== 8. 对比图：全预测 vs 全真实 ====
plt.figure(figsize=(13, 4))
plt.plot(x_series, color='blue', label='True $x_n$')
plt.plot(x_fully_pred, color='red', linestyle='--', label='Fully Predicted $x_n\\prime$')
plt.xlabel('$n$')
plt.ylabel('$x$')
plt.legend(loc='upper right')
plt.text(0.05, 0.9, '(b)', transform=plt.gca().transAxes, fontsize=12)
plt.tight_layout()
plt.savefig('fully_vs_true_sequence.png', dpi=300)  # ← 先保存
plt.show()  # ← 然后展示
print("✅ 图像已保存为 fully_vs_true_sequence.png")

# ==== 9. 保存对比序列 ====
df_full = pd.DataFrame({
    'index': np.arange(N),
    'x_true': x_series,
    'x_full_pred': x_fully_pred
})
df_full.to_csv('logistic_fully_predicted_vs_true.csv', index=False)
print("✅ 序列已保存至 logistic_fully_predicted_vs_true.csv")

# ==== 10. 0-1 Chaos Test: 全预测 vs 全真实 ====
K_fully, M_fully = zero_one_test_for_chaos(x_fully_pred)
print(f"✔️ LSTM 全预测序列的 0-1 chaos K 值：{K_fully:.4f}")

# ==== 11. M(n) 可视化 ====
plt.figure(figsize=(15, 4))
plt.subplot(1, 2, 1)
plt.plot(M_true, label=f"K = {K_true:.3f}")
plt.title("M(n) - True Sequence")
plt.xlabel("n")
plt.ylabel("M(n)")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(M_fully, color='blue', label=f"K = {K_fully:.3f}")
plt.title("M(n) - Fully Predicted")
plt.xlabel("n")
plt.ylabel("M(n)")
plt.legend()

plt.tight_layout()
plt.savefig('Mn_true_vs_fully_predicted.png', dpi=300)
plt.show()
print("✅ 图像已保存为 Mn_true_vs_fully_predicted.png")



# ==== 12. 李雅普诺夫指数估算函数（使用最常见的局部扰动法） ====
from numpy.linalg import norm

def compute_largest_lyapunov(series, m=2, tau=1, eps=1e-6, max_iter=20):
    """
    估算一维时间序列的最大 Lyapunov 指数（Largest Lyapunov Exponent）
    使用相空间重构 + 轨道发散速率分析
    参数:
        series: 1D ndarray, 输入序列
        m: 相空间嵌入维数（embedding dimension）
        tau: 延迟时间
        eps: 初始扰动距离阈值
        max_iter: 每对点比较的最大时间步数
    返回:
        lle: 估算的最大 Lyapunov 指数
    """
    N = len(series) - (m - 1) * tau
    if N <= 0: return np.nan

    # 重构相空间轨道
    X = np.array([series[i:i + (m * tau):tau] for i in range(N)])

    divergence = []
    for i in range(N):
        dists = norm(X - X[i], axis=1)
        dists[i] = np.inf  # 排除自身
        j = np.argmin(dists)
        if dists[j] < eps:
            # 比较未来 max_iter 步的发散程度
            div = []
            for k in range(1, min(max_iter, N - max(i, j))):
                dist_k = norm(X[i + k] - X[j + k])
                if dist_k > 0:
                    div.append(np.log(dist_k))
            if len(div) > 0:
                divergence.append(np.mean(div))

    if len(divergence) == 0:
        return np.nan
    return np.mean(divergence)

lle_true = compute_largest_lyapunov(x_series)
lle_mixed = compute_largest_lyapunov(x_new)
lle_fully = compute_largest_lyapunov(x_fully_pred)

print(f"🌪️ Logistic 原始序列 LLE ≈ {lle_true:.4f}")
print(f"🌪️ LSTM 混合预测序列 LLE ≈ {lle_mixed:.4f}")
print(f"🌪️ LSTM 全预测序列 LLE ≈ {lle_fully:.4f}")


plt.figure(figsize=(8, 5))
labels = ['True', 'Mixed Pred', 'Fully Pred']
values = [lle_true, lle_mixed, lle_fully]
colors = ['blue', 'orange', 'red']

plt.bar(labels, values, color=colors)
plt.ylabel('Largest Lyapunov Exponent (LLE)')
plt.title('LLE Comparison: True vs Predicted')
plt.tight_layout()
plt.savefig('largest_lyapunov_exponent_comparison.png', dpi=300)
plt.show()

print("✅ 图像已保存为 largest_lyapunov_exponent_comparison.png")
