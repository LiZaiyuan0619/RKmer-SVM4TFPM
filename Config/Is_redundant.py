import pandas as pd


def find_and_group_duplicates(csv_path, output_csv_path):
    # 读取CSV文件
    df = pd.read_csv(csv_path)

    # 找出'UNIPROT'列中的重复项
    duplicated = df[df.duplicated('Sequence', keep=False)]

    # 按'UNIPROT'列的值对重复项进行分组
    grouped = duplicated.groupby('Sequence')

    # 创建一个空的DataFrame来收集所有的重复项
    duplicates_collected = pd.DataFrame()

    # 遍历每个组，并将其添加到收集DataFrame中
    for _, group in grouped:
        duplicates_collected = duplicates_collected._append(group)

    # 保存这些重复项到一个新的CSV文件
    duplicates_collected.to_csv(output_csv_path, index=False)
    print(f"Duplicates grouped and saved to {output_csv_path}")



csv_path = './All_Human_TFs.csv'
output_csv_path = 'duplicates5.csv'
find_and_group_duplicates(csv_path, output_csv_path)
