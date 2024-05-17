from src.utils import ROOT_DIR
import sys

sys.path.append(ROOT_DIR)

from src.dataset import CustomDataset
from src.fcm import FCM
from src.fcm import train_epoch, eval_epoch, acc, f1
import torch

import matplotlib.pyplot as plt
from PIL import Image
import requests
import torch
from src.dataset import CustomDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import pandas as pd

import os

# Load the dataset
DATASET_PATH = os.path.join(ROOT_DIR, "data", "MMHS150K", "MMHS150K_with_img_text.csv")
df = pd.read_csv(DATASET_PATH)

# Load the saved predictions

PREDICTIONS_PATH = os.path.join(
    ROOT_DIR, "data", "results_bert_tiny_2", "predictions_bin_bert.json"
)
OUTPUTS_PATH = os.path.join(
    ROOT_DIR, "data", "results_bert_tiny_2", "predictions_prob_bert.json"
)

import json

with open(PREDICTIONS_PATH, "r") as f:
    predictions = json.load(f)
with open(OUTPUTS_PATH, "r") as f:
    outputs = json.load(f)

# For each index, get the binary_hate and label
pred_keys = [int(k) for k in predictions.keys()]

df_test = df[df["index"].isin(pred_keys)]
df_test["pred"] = [predictions[str(k)] for k in df_test["index"]]
df_test["output"] = [outputs[str(k)] for k in df_test["index"]]

# Compare "binary_hate" and "pred" to get the accuracy and F1 score
acc_score = acc(df_test["binary_hate"], df_test["pred"])
f1_score = f1(df_test["binary_hate"], df_test["pred"])

print(f"Accuracy: {acc_score}")
print(f"F1 score: {f1_score}")

# Plot the ROC curve
from sklearn.metrics import roc_curve, auc
import numpy as np

fpr, tpr, _ = roc_curve(df_test["binary_hate"], df_test["output"])
roc_auc = auc(fpr, tpr)

lw = 2
plt.plot(
    fpr, tpr, color="darkorange", lw=lw, label="ROC curve Bert (area = %0.2f)" % roc_auc
)
plt.legend(loc="lower right")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.show()

# Plot the distribution of the outputs / predictions
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
ax[0].hist(df_test["output"], bins=50)
ax[0].set_title("Distribution of the outputs")
ax[0].set_xlabel("Output")
ax[0].set_ylabel("Count")

ax[1].hist(df_test["pred"], bins=50)
ax[1].set_title("Distribution of the predictions")
ax[1].set_xlabel("Prediction")
ax[1].set_ylabel("Count")

plt.show()

# Plot the confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(df_test["binary_hate"], df_test["pred"])

print("True positive rate: ", cm[1, 1] / (cm[1, 1] + cm[1, 0]))
print("True negative rate: ", cm[0, 0] / (cm[0, 0] + cm[0, 1]))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.show()
