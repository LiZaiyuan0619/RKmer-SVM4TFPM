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
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import confusion_matrix, roc_auc_score, make_scorer

logging.basicConfig(level=logging.INFO, filename='logfile_RF.log', filemode='a',
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
            transformed_seq = ''.join(self.raac_dict.get(aa, 'X') for aa in seq)
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


def compute_metrics(y_true, y_prob):
    predicted_labels = np.round(y_prob)
    tn, fp, fn, tp = confusion_matrix(y_true, predicted_labels).ravel()
    acc = (tp + tn) / (tn + tp + fn + fp)
    sen = tp / (tp + fn)
    spe = tn / (tn + fp)
    auc = roc_auc_score(y_true, y_prob)
    mcc_denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if mcc_denominator == 0:
        mcc = 0
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


def custom_score(y_true, y_prob):
    compute_metrics(y_true, y_prob)
    return roc_auc_score(y_true, y_prob)


my_scorer = make_scorer(custom_score, needs_proba=True)

op_list = ["op5", "op8", "op9", "op11",  "op13"]
gap_list = [1, 2, 3, 4, 5]
n_estimators_list = [100, 200, 300]
max_depth_list = [None, 10, 20]

pipe = Pipeline([
    ("transformer", RGDPTransformer()),  # 初始时不指定op和k
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(random_state=114514))
])

param_grid = {
    'transformer__op': op_list,
    'transformer__k': gap_list,
    'rf__n_estimators': n_estimators_list,
    'rf__max_depth': max_depth_list
}

for op in op_list:
    for gap in gap_list:
        logging.info(f"Testing op={op}, gap={gap}")
        pipe = Pipeline([
            ("transformer", RGDPTransformer(op=op, k=gap)),
            ("scaler", StandardScaler()),
            ("rf", RandomForestClassifier(random_state=114514))
        ])

        param_grid = {
            'rf__n_estimators': n_estimators_list,
            'rf__max_depth': max_depth_list,
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
