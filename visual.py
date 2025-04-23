import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
sns.set(style="whitegrid")
plt.rcParams['figure.dpi'] = 120

# 读取一个模型的loss数据
df_x = pd.read_csv('model_x_training_history.csv')

# 创建单个子图
fig, ax = plt.subplots(figsize=(10, 4))
fig.suptitle('LSTM Training Loss Curve', fontsize=16)

# 绘制 x-sequence 模型的曲线
sns.lineplot(data=df_x, x='epoch', y='loss', ax=ax,
             color='#3498db', linewidth=1.5, label='x-sequence')
ax.set_title('x-sequence Model', fontsize=13)
ax.set_xlabel('Training Epoch', fontsize=11)
ax.set_ylabel('MSE Loss', fontsize=11)

# 标记最小 loss
min_loss_x = df_x['loss'].min()
min_epoch_x = df_x[df_x['loss'] == min_loss_x]['epoch'].values[0]
ax.scatter(min_epoch_x, min_loss_x, color='#3498db', s=80, edgecolor='black', zorder=5)
ax.annotate(f'Min: {min_loss_x:.2e}',
            (min_epoch_x, min_loss_x),
            textcoords="offset points",
            xytext=(0, 10),
            ha='center',
            fontsize=10)

# 添加参考线
ax.axhline(y=1e-4, color='gray', linestyle='--', alpha=0.5)
ax.text(x=0.7 * df_x['epoch'].max(), y=1.2e-4,
        s='Reference (1e-4)', color='gray', fontsize=9)

ax.legend(framealpha=0.9)

# 保存高质量图片
plt.tight_layout()
plt.savefig('LSTM_training_loss_x_only.png', bbox_inches='tight', dpi=500)
plt.show()
