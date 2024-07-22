from Bio import SeqIO

def remove_duplicates(large_file, small_file, output_file):
    # 读取小文件中的序列ID
    small_ids = set(record.id for record in SeqIO.parse(small_file, "fasta"))

    # 打开输出文件准备写入
    with open(output_file, 'w') as output_handle:
        # 遍历大文件中的每一条序列
        for record in SeqIO.parse(large_file, "fasta"):
            # 如果该序列的ID不在小文件的ID集合中，写入输出文件
            if record.id not in small_ids:
                SeqIO.write(record, output_handle, "fasta")

# 使用示例
large_file = "D:/ML/ProBERT/All_Human_TFs.txt"
small_file = "merged_myTFs.txt"
output_file = "Predict.txt"

remove_duplicates(large_file, small_file, output_file)
