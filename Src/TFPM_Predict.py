import itertools
import numpy as np
import pandas as pd
import math
import scipy.stats
import timeit
from Bio import SeqIO
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
import logging

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logging.basicConfig(level=logging.INFO, filename='logfile_Predict.log', filemode='a',  # 修改为 'a' 以续写日志
                    format='%(asctime)s - %(levelname)s - %(message)s')


class RGDPTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, op='op13', k=3):
        self.k = k
        self.op = op
        self.raac_dict = None
        self.groups = None

    def fit(self, X, y=None):
        RAAC_scheme = {
            'op5': [['G'], ['I', 'V', 'F', 'Y', 'W'], ['A', 'L', 'M', 'E', 'Q', 'R', 'K'],
                    ['P'], ['N', 'D', 'H', 'S', 'T', 'C']],
            'op8': [['G'], ['I', 'V'], ['F', 'Y', 'W'], ['A', 'L', 'M'], ['E', 'Q', 'R', 'K'],
                    ['P'], ['N', 'D'], ['H', 'S', 'T', 'C']],
            'op9': [['G'], ['I', 'V'], ['F', 'Y', 'W'], ['A', 'L', 'M'], ['E', 'Q', 'R', 'K'],
                    ['P'], ['N', 'D'], ['H', 'S'], ['T', 'C']],
            'op11': [['G'], ['I', 'V'], ['F', 'Y', 'W'], ['A'], ['L', 'M'], ['E', 'Q', 'R', 'K'], ['P'], ['N', 'D'],
                     ['H', 'S'], ['T'], ['C']],
            'op13': [['G'], ['I', 'V'], ['F', 'Y', 'W'], ['A'], ['L'], ['M'], ['E'], ['Q', 'R', 'K'], ['P'], ['N', 'D'],
                     ['H', 'S'], ['T'], ['C']],
            'op20': ['G', 'I', 'V', 'F', 'Y', 'W', 'A', 'L', 'M', 'E', 'Q', 'R', 'K', 'P', 'N', 'D', 'H', 'S', 'T', 'C']
        }
        self.groups = RAAC_scheme[self.op]
        self.raac_dict = {amino_acid: group[0] for group in self.groups for amino_acid in group}
        return self

    def transform(self, X, y=None):
        def generate_kmers(k, alphabet):
            if k == 1:
                return alphabet
            else:
                subkmers = generate_kmers(k - 1, alphabet)
                return [x + y for x in alphabet for y in subkmers]
        alphabet = [group[0] for group in self.groups]
        kmers = generate_kmers(self.k, alphabet)
        kmer_dict = {kmer: 0 for kmer in kmers}

        result = []
        for seq in X:
            transformed_seq = ''.join(self.raac_dict.get(aa, 'X') for aa in seq)  # 'X'用于未定义的字符
            current_kmer_counts = kmer_dict.copy()

            for i in range(len(transformed_seq) - self.k + 1):
                kmer = transformed_seq[i:i + self.k]
                if kmer in current_kmer_counts:
                    current_kmer_counts[kmer] += 1

            total_kmers = sum(current_kmer_counts.values())
            normalized_kmer_counts = [count / total_kmers for count in current_kmer_counts.values()]

            result.append(normalized_kmer_counts)

        return np.array(result)


def read_fasta(path, maxlen=1500, encode='token'):
    fasta_sequences = SeqIO.parse(open(path), 'fasta')
    sequences = []
    labels = []
    for fasta in fasta_sequences:
        name, sequence = str(fasta.id), str(fasta.seq)
        sequences.append(sequence)
        labels.append(name)

    if encode == 'token':
        return sequences, labels

    return None

# predct_path = 'Data/All_Human_TFs.txt'
predct_path = 'Data/Ori_522TFs.txt'

# 读取新的预测文件
extra_sequences, extra_labels = read_fasta(predct_path)

# 使用最佳参数组合创建SVM模型管道
op_list = ["op11"]
gap_list = [1]
gamma_list = [0.001]
C_list = [0.01]

pipe = Pipeline([
    ("transformer", RGDPTransformer(op=op_list[0], k=gap_list[0])),
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True, gamma=gamma_list[0], C=C_list[0], random_state=114514))
])

# 在训练数据上训练模型
TFPM_used, _ = read_fasta('Data/Ori_Data/TFPM_training_dataset_used.txt')
TFPM_unused, _ = read_fasta('Data/Ori_Data/TFPM_training_dataset_unused.txt')
TFPNM, _ = read_fasta('Data/Ori_Data/TFPNM_training_dataset.txt')

X_train = TFPNM + TFPM_used + TFPM_unused
y_train = np.append([np.zeros(len(TFPNM), dtype=np.int64)],
                    [np.ones(len(TFPM_used) + len(TFPM_unused), dtype=np.int64)])

pipe.fit(X_train, y_train)

# 在新的数据集上进行预测
extra_probs = pipe.predict_proba(extra_sequences)[:, 1]

# 找出概率最高的20个样本的索引
top_20_indices = extra_probs.argsort()[-20:][::-1]

# 获取这些样本的名称和概率
top_20_sample_names = [extra_labels[i] for i in top_20_indices]
top_20_sample_probs = [extra_probs[i] for i in top_20_indices]

# 输出结果
print("Top 20 samples with highest TFPM probability:")
logging.info(f"Top 20 samples with highest TFPM probability:{predct_path}")
for name, prob in zip(top_20_sample_names, top_20_sample_probs):
    print(f"{name}: {prob:.4f}")
    logging.info(f"{name}: {prob:.4f}")
logging.info("=================End==================")