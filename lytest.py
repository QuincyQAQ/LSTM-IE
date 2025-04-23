import numpy as np
import matplotlib.pyplot as plt

#==== Logistic 映射函数 ====
def logistic_map(r, x0, N):
    x = [x0]
    for _ in range(N - 1):
        x.append(r * x[-1] * (1 - x[-1]))
    return np.array(x)

# def logistic_map(a, x0, N, eps=1e-6):
#     x = [x0]
#     for _ in range(N - 1):
#         denominator = np.sin(x[-1])
#         # 避免除以 0 或极小数
#         if abs(denominator) < eps:
#             denominator = eps if denominator >= 0 else -eps
#         x_next = np.sin(a / denominator)
#         x.append(x_next)
#     return np.array(x)



# ==== 伪预测函数（模拟混合预测效果） ====
def mixed_sequence(original, pred_len):
    """
    构造一半原始、一半伪预测混合序列（这里用扰动模拟预测）
    """
    mix = np.copy(original)
    noise = np.random.normal(0, 0.02, size=pred_len)  # 加一点噪声模拟LSTM误差
    mix[-pred_len:] += noise
    return mix

# ==== 0-1 chaos test（Lyapunov 映射） ====
def zero_one_test_for_chaos(x, c=0.7, normalize=True):
    if normalize:
        x = (x - np.mean(x)) / np.std(x)
    n = len(x)
    t = np.arange(1, n + 1)
    cs = np.cumsum(x * np.cos(c * t))
    ss = np.cumsum(x * np.sin(c * t))
    M = cs**2 + ss**2
    return np.corrcoef(t, M)[0, 1]

# ==== 参数设置 ====
a_values = np.linspace(2.5, 4.0, 50)
K_true_list = []
K_mixed_list = []

N = 2000
train_len = N // 2

# ==== 计算 Lyapunov 映射测试图 ====
for a in a_values:
    x_series = logistic_map(a, x0=0.1, N=N)
    x_mixed = mixed_sequence(x_series, pred_len=N - train_len)

    K_true = zero_one_test_for_chaos(x_series)
    K_mixed = zero_one_test_for_chaos(x_mixed)

    K_true_list.append(K_true)
    K_mixed_list.append(K_mixed)

# ==== 绘图 ====
plt.figure(figsize=(8, 5))
plt.plot(a_values, K_true_list, label='True Logistic Sequence', marker='o')
plt.plot(a_values, K_mixed_list, label='Half True + Half Pseudo Prediction', marker='s')
plt.xlabel("Logistic Map Parameter $a$")
plt.ylabel("0-1 Chaos $K$ Value")
plt.title("Lyapunov-like 0-1 Chaos Test across Parameter $a$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
