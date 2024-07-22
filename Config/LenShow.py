
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 读取CSV文件
file_path = 'D:/ML/ProBERT/myAMP2/TF_Train.csv'  # 请将此路径替换为你的xlsx文件的实际路径

df = pd.read_csv(file_path)
# 获取aa_len列的值
aa_lengths = df['aa_len']

# 绘制直方图
plt.figure(figsize=(10, 6))
plt.hist(aa_lengths, bins=50, range=(0, 5000), color='skyblue', edgecolor='black', density=False, cumulative=False, label='Frequency')

# 添加累积百分比折线图
sorted_aa_lengths = np.sort(aa_lengths)
total_length = float(len(sorted_aa_lengths))
percentile_values = [len(sorted_aa_lengths[sorted_aa_lengths <= i]) / total_length * 100 for i in range(5001)]
plt.twinx()
plt.plot(range(5001), percentile_values, color='red', label='Cumulative Percentage')

# 标记长度为1000的位置
plt.axvline(x=1000, color='green', linestyle='--')
plt.text(1000, 50, 'Length 1000', rotation=0, verticalalignment='bottom')  # 调整文本的方向

# 标记交点处的累积百分比
percentile_at_1000 = len(sorted_aa_lengths[sorted_aa_lengths <= 1000]) / total_length * 100
plt.text(1000, percentile_at_1000, f'{percentile_at_1000:.2f}%', color='blue', rotation=0, verticalalignment='top')  # 调整文本的方向

# 设置图例
plt.legend(loc='upper right')

# 设置标题和标签
plt.title('Protein Length Distribution with Cumulative Percentage')
plt.xlabel('Length')
plt.ylabel('Frequency / Cumulative Percentage')

# 设置x轴刻度
plt.xticks(np.arange(0, 5001, 500))

# 显示网格
plt.grid(True)
# 保存图像
plt.savefig('protein_length_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

