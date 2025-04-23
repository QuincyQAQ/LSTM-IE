import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim


# ==== Logistic 映射函数 ====
def logistic_map(r, x0, N):
    x = [x0]
    for _ in range(N - 1):
        x_next = r * x[-1] * (1 - x[-1])
        x.append(x_next)
    return np.array(x)


# ==== LSTM 相关函数 ====
class BiLSTMForecast(nn.Module):
    def __init__(self, hidden_size):
        super(BiLSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size=1,
                            hidden_size=hidden_size,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        return self.fc(last_out)


def prepare_data(series, train_len, seq_len, device):
    X, Y = [], []
    for i in range(train_len - seq_len):
        X.append(series[i:i + seq_len])
        Y.append(series[i + seq_len])
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    Y = torch.tensor(Y, dtype=torch.float32).unsqueeze(-1).to(device)
    return X, Y


def train_model(X_train, Y_train, hidden_size=64, epochs=300, lr=1e-2):
    model = BiLSTMForecast(hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_train)
        loss = criterion(output, Y_train)
        loss.backward()
        optimizer.step()
    return model


def predict_future(model, series, train_len, pred_len, seq_len, device):
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


# ==== 0-1 chaos test ====
def zero_one_test_for_chaos(x, c=0.7, normalize=True):
    if normalize:
        x = (x - np.mean(x)) / np.std(x)
    n = len(x)
    t = np.arange(1, n + 1)
    cs = np.cumsum(x * np.cos(c * t))
    ss = np.cumsum(x * np.sin(c * t))
    M = cs ** 2 + ss ** 2
    return np.corrcoef(t, M)[0, 1]


# ==== 扫描参数 a 的范围并计算 K 值 ====
a_values = np.linspace(3.5, 4.0, 30)
K_true_list = []
K_pred_list = []

N = 1000
train_len = 500
seq_len = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

for a in a_values:
    x_series = logistic_map(a, x0=0.1, N=N)
    X_train, Y_train = prepare_data(x_series, train_len, seq_len, device)
    model = train_model(X_train, Y_train)
    x_pred = predict_future(model, x_series, train_len, N - train_len, seq_len, device)

    K_true = zero_one_test_for_chaos(x_series)
    K_pred = zero_one_test_for_chaos(x_pred)

    K_true_list.append(K_true)
    K_pred_list.append(K_pred)

# ==== 绘图 ====
plt.figure(figsize=(8, 5))
plt.plot(a_values, K_true_list, label='True Chaos (Full Logistic)', marker='o')
plt.plot(a_values, K_pred_list, label='LSTM Mixed Prediction', marker='s')
plt.xlabel("Logistic Map Parameter $a$")
plt.ylabel("0-1 Chaos $K$ Value")
plt.title("Lyapunov-like 0-1 Test for Logistic Map and LSTM Prediction")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
