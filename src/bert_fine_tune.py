import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertModel,
    BertTokenizer,
)

from src.utils import ROOT_DIR


def load_pretrained(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    if torch.cuda.is_available():
        model = model.to("cuda:0")
    return tokenizer, model


# load bert
tokenizer, model = load_pretrained("bert-base-cased")

# loading dataset
print(ROOT_DIR)
tweets = pd.read_csv("data/MMHS150K/MMHS150K_with_img_text.csv")
data = tweets[["tweet_text_clean", "binary_hate", "split"]]
data.head()

# splitting data
X_train = data[data["split"] == "train"]["tweet_text_clean"].reset_index(drop=True)
X_val = data[data["split"] == "val"]["tweet_text_clean"].reset_index(drop=True)
X_test = data[data["split"] == "test"]["tweet_text_clean"].reset_index(drop=True)

y_train = data[data["split"] == "train"]["binary_hate"].reset_index(drop=True)
y_val = data[data["split"] == "val"]["binary_hate"].reset_index(drop=True)
y_test = data[data["split"] == "test"]["binary_hate"].reset_index(drop=True)

# tokenizing
X_train_tokenized = tokenizer(
    X_train.astype(str).tolist(), padding=True, truncation=True, max_length=512
)
X_val_tokenized = tokenizer(
    X_val.astype(str).tolist(), padding=True, truncation=True, max_length=512
)
X_test_tokenized = tokenizer(
    X_test.astype(str).tolist(), padding=True, truncation=True, max_length=512
)

# create iterator for training
from torch.utils.data import DataLoader, Dataset


class MMHSDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


train_dataset = MMHSDataset(X_train_tokenized, y_train)
val_dataset = MMHSDataset(X_val_tokenized, y_val)
test_dataset = MMHSDataset(X_test_tokenized, y_test)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)


def compute_metrics(pred):
    pred, labels = pred
    preds = np.argmax(pred, axis=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}


# training
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./results",  # output directory
    num_train_epochs=10,  # total number of training epochs
    per_device_train_batch_size=8,  # batch size per device during training
    report_to="none",
    learning_rate=0.001,
)

trainer = Trainer(
    model=model,  # the instantiated 🤗 Transformers model to be trained
    args=training_args,  # training arguments, defined above
    train_dataset=train_dataset,  # training dataset
    eval_dataset=val_dataset,  # evaluation dataset
    compute_metrics=compute_metrics,
)

# train
trainer.train()

# Evaluate the model
eval_results = trainer.evaluate()

# Print the evaluation results
for key, value in eval_results.items():
    print(f"{key}: {value}")

trainer.save_model("BertFineTuned")
