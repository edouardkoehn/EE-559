from src.dataset import CustomDataset
from src.utils import ROOT_DIR

import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import seaborn as sns
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


def get_Lava_results_1(path_results, test_data):
    with open(path_results, "r") as f:
        lava_prediction = json.load(f)

    lava_prediction = convert_label2id(lava_prediction)
    index = np.array([int(key) for key, val in lava_prediction.items()])
    prediction = np.array([int(val) for key, val in lava_prediction.items()])

    lava_prediction = np.concatenate(
        [np.expand_dims(index, axis=1), np.expand_dims(prediction, axis=1)], axis=1
    )
    return lava_prediction


def find_intersection_set(index_1, index_2, index_3, index_4):
    set1 = set(index_1)
    set2 = set(index_2)
    set3 = set(index_3)
    set4 = set(index_4)
    inter = set1.intersection(set2)
    inter = inter.intersection(set3)
    inter = inter.intersection(set4)
    return list(inter)


def compute_matrix_data(
    index_set,
    matrix_1,
    matrix_2,
    matrix_3,
    matrix_4,
    prompt1,
    prompt2,
    prompt3,
    prompt4,
):
    """Simple method for extracting the value of matrix_1[:,1] that"""
    p1 = matrix_1[np.isin(matrix_1[:, 0], index_set), :][:, 1]
    p2 = matrix_2[np.isin(matrix_2[:, 0], index_set), :][:, 1]
    p3 = matrix_3[np.isin(matrix_3[:, 0], index_set), :][:, 1]
    p4 = matrix_4[np.isin(matrix_4[:, 0], index_set), :][:, 1]

    matrix = np.column_stack((index_set, p1, p2, p3, p4))
    # df=pd.DataFrame(data=matrix.T)
    df = pd.DataFrame(
        data=matrix[:, 1:],
        columns=[prompt1, prompt2, prompt3, prompt4],
        index=matrix[:, 0],
    )
    return df


def get_level_agrement_dataset(index_set, dataset):
    return [
        {idx: dataset.get_data_from_index(int(idx))["hate_speech"].values[0]}
        for idx in index_set
    ]


# load the data
transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
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

## Compute the level of agreement
path = os.path.join(
    ROOT_DIR, "data", "results", "Llava", "Lava0S_prediction_on_test_final.json"
)
pred_lava_image_only = get_Lava_results_1(path, test_data)

path = os.path.join(
    ROOT_DIR,
    "data",
    "results",
    "Llava",
    "Lava0S_tweet_only_prediction_on_test_final.json",
)
pred_laval_tweet_only = get_Lava_results_1(path, test_data)

path = os.path.join(
    ROOT_DIR, "data", "results", "Llava", "Lava0S_tweet_prediction_on_test_final.json"
)
pred_laval_image_and_tweet = get_Lava_results_1(path, test_data)

path = os.path.join(
    ROOT_DIR,
    "data",
    "results",
    "Llava",
    "Lava0S_tweet_informed_prediction_on_test_final.json",
)
pred_laval_image_and_tweet_informed = get_Lava_results_1(path, test_data)

index_set = find_intersection_set(
    pred_lava_image_only[:, 0],
    pred_laval_tweet_only[:, 0],
    pred_laval_image_and_tweet[:, 0],
    pred_laval_image_and_tweet_informed[:, 0],
)

matrix = compute_matrix_data(
    index_set,
    pred_lava_image_only,
    pred_laval_tweet_only,
    pred_laval_image_and_tweet,
    pred_laval_image_and_tweet_informed,
    "Image_only",
    "Tweet_only",
    "Image_Tweet",
    "Image_Tweet_informed",
)

# Assuming 'matrix' is your DataFrame
correlation_matrix = matrix.corr("spearman")
mask = np.array(
    [
        [False, True, True, True],
        [False, False, True, True],
        [False, False, False, True],
        [False, False, False, False],
    ]
)
hue_neg, hue_pos = 250, 15
cmap = sns.diverging_palette(hue_neg, hue_pos, center="dark", as_cmap=True)
# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(
    correlation_matrix,
    mask=mask,
    annot=True,
    cmap=cmap,
    vmin=-1,
    vmax=1,
    xticklabels=matrix.columns,
    yticklabels=matrix.columns,
)
plt.title("Level of agreement between prompt")
plt.savefig(os.path.join(ROOT_DIR, "data", "figure", f"llava_level_agreement.png"))
plt.show()
