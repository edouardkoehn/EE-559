from src.utils import ROOT_DIR
import sys


from src.fcm import acc, f1
import matplotlib.pyplot as plt
import pandas as pd
import os
from sklearn.metrics import roc_curve, auc
import json
from sklearn.metrics import confusion_matrix
import seaborn as sns

##Simple script to generate the figure of the FCM model

# Load the dataset
DATASET_PATH = os.path.join(ROOT_DIR, "data", "MMHS150K", "MMHS150K_with_img_text.csv")
df = pd.read_csv(DATASET_PATH)

# Load the saved predictions
PREDICTIONS_PATH = os.path.join(
    ROOT_DIR, "data", "results", "FCM", "fcm_predictions.json"
)
OUTPUTS_PATH = os.path.join(ROOT_DIR, "data", "results", "FCM", "fcm_outputs.json")
FIG_OUTPUT_PATH = os.path.join(ROOT_DIR, "data", "figure")

if not os.path.exists(FIG_OUTPUT_PATH):
    os.mkdir(FIG_OUTPUT_PATH)

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

print(f"FCM model_Accuracy: {acc_score}")
print(f"FCM model_F1 score: {f1_score}")

# Plot the ROC curve
fpr, tpr, _ = roc_curve(df_test["binary_hate"], df_test["output"])
roc_auc = auc(fpr, tpr)

lw = 2
plt.plot(
    fpr, tpr, color="darkorange", lw=lw, label="ROC curve FCM (area = %0.2f)" % roc_auc
)
plt.legend(loc="lower right")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC curve")
plt.savefig(os.path.join(FIG_OUTPUT_PATH, "fcm_ROC_.png"))
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
plt.savefig(os.path.join(FIG_OUTPUT_PATH, "fcm_distribution_.png"))
plt.show()
# Plot the confusion matrix
cm = confusion_matrix(df_test["binary_hate"], df_test["pred"])

print("FCM model_True positive rate: ", cm[1, 1] / (cm[1, 1] + cm[1, 0]))
print("FCM model_True negative rate: ", cm[0, 0] / (cm[0, 0] + cm[0, 1]))

sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted labels")
plt.ylabel("True labels")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(FIG_OUTPUT_PATH, "fcm_confusion_matrix_.png"))
plt.show()
