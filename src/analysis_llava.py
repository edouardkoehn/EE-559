from src.dataset import CustomDataset
from src.utils import ROOT_DIR

import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
import json


def convert_label2id(res_dict):
    """Simple method to convert the label from string to int"""
    label2id = {"not hateful": 0, "hateful": 1}
    for key, val in res_dict.items():
        res_dict[key] = label2id[val]
    return res_dict


def convert_id2label(res_dict):
    """Simple method to convert the label from int to string"""
    id2label = {0: "not hateful", 1: "hateful"}
    for key, val in res_dict.items():
        res_dict[key] = id2label[val]
    return res_dict


def find_label(res_dict, dataset):
    """simple method to align the label with the prediction"""
    y_pred = [val for key, val in res_dict.items()]
    y_label = [
        dataset.get_data_from_index(int(key))["binary_hate"].values[0]
        for key, val in res_dict.items()
    ]
    return y_pred, y_label


def compute_confusion_matrix(y_pred, y_label):
    tn, fp, fn, tp = confusion_matrix(y_label, y_pred).ravel()
    return tn, fp, fn, tp


def compute_accuracy(y_pred, y_label):
    return accuracy_score(y_label, y_pred, normalize=True)


def plot_confusion_matrix(model_name: str, y_pred, y_label):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true=y_label,
        y_pred=y_pred,
        display_labels=["not hateful", "hateful"],
        cmap="Blues",
    )
    plt.title("Confusion Matrix")
    plt.savefig(
        os.path.join(
            ROOT_DIR, "data", "figure", f"llava_{model_name}_confusion_matrix.png"
        )
    )
    plt.show()


def get_Lava_results(path_results, model_name, test_data):
    with open(path_results, "r") as f:
        lava_prediction = json.load(f)
    lava_prediction = convert_label2id(lava_prediction)
    lava_pred, lava_label = find_label(lava_prediction, test_data)
    print(
        f"Accuracy_{model_name}:",
        "{:.3f}".format(compute_accuracy(lava_pred, lava_label)),
    )
    plot_confusion_matrix(model_name, lava_pred, lava_label)


transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
# load the data
test_data = CustomDataset(
    csv_file=os.path.join(ROOT_DIR, "data", "MMHS150K", "MMHS150K_with_img_text.csv"),
    img_dir=os.path.join(ROOT_DIR, "data", "MMHS150K", "img_resized/"),
    split="test",
    transform=transform,
)

# Lava image only
path_lava_image_prediction = os.path.join(
    ROOT_DIR, "data", "results", "Llava", "Lava0S_prediction_on_test_final.json"
)
get_Lava_results(path_lava_image_prediction, "Llava Image only", test_data)
# Lava tweet only
path_lava_tweet_only_prediction = os.path.join(
    ROOT_DIR,
    "data",
    "results",
    "Llava",
    "Lava0S_tweet_only_prediction_on_test_final.json",
)
get_Lava_results(path_lava_tweet_only_prediction, "Llava tweet only", test_data)
# Lava image +tweet
path_lava_image_tweet_prediction = os.path.join(
    ROOT_DIR, "data", "results", "Llava", "Lava0S_tweet_prediction_on_test_final.json"
)
get_Lava_results(path_lava_image_tweet_prediction, "Llava Image + Tweet", test_data)
# Lava image + tweet informed
path_lava_image_tweet_informed_prediction = os.path.join(
    ROOT_DIR,
    "data",
    "results",
    "Llava",
    "Lava0S_tweet_informed_prediction_on_test_final.json",
)
get_Lava_results(
    path_lava_image_tweet_informed_prediction,
    "Llava Image + tweet with informed prompt",
    test_data,
)
