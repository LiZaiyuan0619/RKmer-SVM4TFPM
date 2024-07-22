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

logging.basicConfig(level=logging.INFO, filename='logfile_RF.log', filemode='a',  # 修改为 'a' 以续写日志
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
        # 递归函数，导致内存占用很大，生成所有可能的k-mer序列
        # 当 k == 1 时，直接返回字母表
        # 当 k > 1 时，递归生成长度为 k-1 的子序列，然后将字母表中的每个字母与子序列拼接
        def generate_kmers(k, alphabet):
            if k == 1:
                return alphabet
            else:
                subkmers = generate_kmers(k - 1, alphabet)
                return [x + y for x in alphabet for y in subkmers]
        # alphabet 是RAAC分组中的第一个元素组成的列表
        alphabet = [group[0] for group in self.groups]
        # kmers 调用 generate_kmers 函数生成所有可能的k-mer
        kmers = generate_kmers(self.k, alphabet)
        # kmer_dict 是一个字典，键是所有可能的k-mer，值是0
        kmer_dict = {kmer: 0 for kmer in kmers}

        result = []
        for seq in X:
            # 应用RAAC方案
            # transformed_seq 将原始序列中的每个氨基酸替换为其对应的RAAC分组中的第一个元素。如果某个氨基酸不在 raac_dict 中，则替换为 'X
            transformed_seq = ''.join(self.raac_dict.get(aa, 'X') for aa in seq)  # 'X'用于未定义的字符

            # 初始化k-mer计数
            # current_kmer_counts 初始化为 kmer_dict 的副本
            current_kmer_counts = kmer_dict.copy()

            for i in range(len(transformed_seq) - self.k + 1):
                kmer = transformed_seq[i:i + self.k]
                if kmer in current_kmer_counts:
                    current_kmer_counts[kmer] += 1

            # 归一化k-mer频率
            total_kmers = sum(current_kmer_counts.values())
            # 即该k-mer在序列中出现的次数除以总k-mer数
            normalized_kmer_counts = [count / total_kmers for count in current_kmer_counts.values()]

            result.append(normalized_kmer_counts)

        return np.array(result)


def read_fasta(path, maxlen=1500, encode='token'):
    # 读取FASTA文件，返回一个序列迭代器 fasta_sequences
    fasta_sequences = SeqIO.parse(open(path), 'fasta')
    sequences = []
    labels = []
    # name 是序列的ID（标签），sequence 是实际的核酸或氨基酸序列
    # 将 sequence 和 name 分别添加到 sequences 和 labels 列表中
    for fasta in fasta_sequences:
        name, sequence = str(fasta.id), str(fasta.seq)
        sequences.append(sequence)
        labels.append(name)

    if encode == 'token':
        return sequences, labels

    return None


def compute_metrics(y_true, y_prob):
    predicted_labels = np.round(y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted_labels).ravel()
    acc = (tp + tn) / (tn + tp + fn + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    auc = roc_auc_score(y_true, y_prob)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if mcc_denominator == 0:
        mcc = 0  # 或者设置为NaN: float('nan')
    else:
        mcc = (tp * tn - fp * fn) / mcc_denominator
    f1_score = 2 * tp / (2 * tp + fp + fn)
    logging.info("acc,       sen,       spe,      auc,      mcc,     f1_score")
    logging.info("%0.7s,  %0.7s,  %0.7s,  %0.7s,  %0.7s,  %0.7s", acc, sen, spe, auc, mcc, f1_score)

    return acc, sen, spe, auc, mcc, f1_score


TFPM_used, _ = read_fasta('data/TFPM_training_dataset_used.txt')
TFPM_unused, _ = read_fasta('data/TFPM_training_dataset_unused.txt')
TFPNM, _ = read_fasta('data/TFPNM_training_dataset.txt')

X_train = TFPNM + TFPM_used + TFPM_unused
y_train = np.append([np.zeros(len(TFPNM), dtype=np.int64)],
                    [np.ones(len(TFPM_used) + len(TFPM_unused), dtype=np.int64)])

TFPM_test, _ = read_fasta('data/TFPM_independent_dataset.txt')
TFPNM_test, _ = read_fasta('data/TFPNM_independent_dataset.txt')

X_test = TFPNM_test + TFPM_test
y_test = np.append([np.zeros(len(TFPNM_test), dtype=np.int64)], [np.ones(len(TFPM_test), dtype=np.int64)])


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, average_precision_score, roc_auc_score, accuracy_score, matthews_corrcoef
from sklearn.metrics import make_scorer



def custom_score(y_true, y_prob):
    compute_metrics(y_true, y_prob)
    return roc_auc_score(y_true, y_prob)


my_scorer = make_scorer(custom_score, needs_proba=True)

# op_list = ["op13"]
# gap_list = [1, 2, 3, 4, 5]
# gamma_list = [10, 1, 0.1, 0.01, 0.001]
# C_list = [0.001, 0.01, 0.1, 1, 10]
op_list = ["op13"]
gap_list = [1]
gamma_list = [0.001]
C_list = [ 0.01]

pipe = Pipeline([
    ("transformer", RGDPTransformer()),  # 初始时不指定op和k
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True, random_state=114514))
])

param_grid = {
    'transformer__op': op_list,
    'transformer__k': gap_list,
    'svm__gamma': gamma_list,
    'svm__C': C_list
}


for op in op_list:
    for gap in gap_list:
        logging.info(f"Testing op={op}, gap={gap}")
        pipe = Pipeline([
            ("transformer", RGDPTransformer(op=op, k=gap)),
            ("scaler", StandardScaler()),
            ("svm", SVC(probability=True, random_state=114514))
        ])

        param_grid = {
            'svm__gamma': gamma_list,
            'svm__C': C_list,
        }

        grid_search = GridSearchCV(pipe, param_grid, scoring=my_scorer, cv=RepeatedKFold(n_splits=5, n_repeats=1, random_state=114514), n_jobs=4, verbose=3)
        grid_search.fit(X_train, y_train)

        optimal_auc = grid_search.best_score_
        optimal_params = grid_search.best_params_
        optimal_pipe = grid_search.best_estimator_

        logging.info("Optimal AUC: %s", optimal_auc)
        logging.info("Best parameters: %s", optimal_params)

        y_test_probs = optimal_pipe.predict_proba(X_test)[:, 1]
        logging.info("Metrics on test set:")
        compute_metrics(y_test, y_test_probs)
        logging.info("---------------End------------------")