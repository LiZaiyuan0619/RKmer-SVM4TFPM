import os
import pandas as pd
import torch

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, matthews_corrcoef, \
    roc_auc_score

torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com/"
from transformers import AutoTokenizer, Trainer, TrainingArguments, BertForSequenceClassification, AdamW


class amp_data():
    def __init__(self, df, tokenizer_name='Rostlab/prot_bert_bfd', max_len=1000):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, do_lower_case=False)
        self.max_len = max_len
        self.seqs, self.labels = self.get_seqs_labels()

    def get_seqs_labels(self):
        seqs = list(df['aa_seq'])
        labels = list(df['AMP'].astype(int))

        assert len(seqs) == len(labels)
        return seqs, labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        seq = " ".join("".join(self.seqs[idx].split()))
        seq_ids = self.tokenizer(seq, truncation=True, padding='max_length', max_length=self.max_len)
        sample = {key: torch.tensor(val) for key, val in seq_ids.items()}
        sample['labels'] = torch.tensor(self.labels[idx])

        return sample


# data_path = './enhanced_dataset.csv'
# data_path = './91RateData/TFPM_Train.csv'
# data_path = './myAMP2/TFPM_Train_unBalanced.csv'
data_path = './myAMP2/TF_Train.csv'
df = pd.read_csv(data_path, index_col = 0)

df = df.sample(frac=1, random_state = 114514)
train_dataset = amp_data(df)
print("Training dataset created...")
print("Training dataset shape: ", df.shape)


# data_path = './91RateData/TFPM_independent.csv'
data_path = './myAMP2/TF_independent.csv'

df = pd.read_csv(data_path, index_col = 0)
df = df.sample(frac=1, random_state = 114514)
eval_dataset = amp_data(df)
print("Evaluation dataset created...")
print("Evaluation dataset shape: ", df.shape)

# # define the necessary metrics for performance evaluation
# # 在模型训练和评估过程中计算一系列性能指标，包括精确度（precision）、召回率（recall）、F1 分数和准确率（accuracy）
# def compute_metrics(pred):
#     labels = pred.label_ids
#     preds = pred.predictions.argmax(-1)
#     precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
#     acc = accuracy_score(labels, preds)
# #     conf = confusion_matrix(labels, preds)
#     return {
#         'accuracy': acc,
#         'f1': f1,
#         'precision': precision,
#         'recall': recall,
# #         'confusion matrix': conf
#     }

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    acc = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)
    mcc = matthews_corrcoef(labels, preds)
    f1_score = precision_recall_fscore_support(labels, preds, average='binary')[2]

    auc = roc_auc_score(labels, pred.predictions[:, 1]) if pred.predictions.ndim > 1 else None

    return {
        'accuracy': acc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'MCC': mcc,
        'AUC': auc,
        "f1_score": f1_score
    }


def model_init():
    return BertForSequenceClassification.from_pretrained('Rostlab/prot_bert_bfd')

print("Training arguments set...")
training_args = TrainingArguments(
    output_dir='./results_oup',
    num_train_epochs=5,
    learning_rate = 5e-5,
    per_device_train_batch_size=2,
    warmup_steps=0,
    weight_decay=0.1,
    logging_dir='./logs',
    logging_steps=10,
    do_train=True,
    do_eval=True,
    evaluation_strategy="steps",
    eval_steps=10,
    save_strategy='steps',
    gradient_accumulation_steps=10,
    fp16=True,
    fp16_opt_level="O2",
    run_name="AMP-BERT",
    seed=0,
    load_best_model_at_end = True
)
print("training arguments ", training_args)

print("Trainer set...")
trainer = Trainer(
    model=model_init(),
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)
print('Training...')
trainer.train()
print('Training finished...')
# trainer.save_model('./results_save/')
# print('Model saved...')
