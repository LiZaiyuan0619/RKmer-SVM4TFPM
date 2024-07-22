import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'D:/Desktop/XGBoost.xlsx'  # Update the path if necessary
df = pd.read_excel(file_path, index_col=0)

# 绘制热图
plt.figure(figsize=(10, 8))
heatmap = sns.heatmap(df, annot=True, fmt=".4f", cmap="Reds", cbar_kws={'label': 'Scale'}, linewidths=.5)

# 突出显示最大值
max_value = df.max().max()
max_value_position = df.stack().idxmax()

# 确保索引是整数类型
row_index = df.index.get_loc(max_value_position[0])
col_index = df.columns.get_loc(max_value_position[1])

# 添加矩形边框
heatmap.add_patch(plt.Rectangle((col_index, row_index), 1, 1, fill=False, edgecolor='yellow', lw=3))

# 添加标题和标签
plt.title('Best AUC values from repeated fivefold cross-validation for different settings of reduced K-mer')
plt.xlabel('op')
plt.ylabel('K')

# 保存图像
plt.savefig('heatmapXGBoost.png', dpi=400, bbox_inches='tight')

# 显示图表
plt.show()