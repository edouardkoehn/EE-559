import json
import os
import re

import numpy as np
import pandas as pd

from src.utils import ROOT_DIR

"""Simple script to reformat the data"""

## Will load the text-side of the dataset, put it in a pandas dataframe and save it in a csv file
# for ease of use in the future

# Data folder
DATA_FOLDER = os.path.join(ROOT_DIR, "data", "MMHS150K", "MMHS150K_GT.json")
# Folder with the img_txt
IMG_TEXT_FOLDER = os.path.join(ROOT_DIR, "data", "MMHS150K", "img_txt/")
# Splits folder
SPLITS_FOLDER = os.path.join(ROOT_DIR, "data", "MMHS150K", "splits/")
# Output csv files
OUTPUT_CSV = os.path.join(ROOT_DIR, "data", "MMHS150K", "MMHS150K.csv")
OUTPUT_TEXT_IN_IMAGE = os.path.join(
    ROOT_DIR, "data", "MMHS150K", "MMHS150K_text_in_image.csv"
)
## Load data
data = pd.read_json(
    DATA_FOLDER, orient="index", convert_dates=False, convert_axes=False
)
data = data.reset_index(drop=False)
data["index"] = data["index"].astype(int)

## Clean the tweet text
# Keep only the text before https://t.co/
data["tweet_text_clean"] = data["tweet_text"].str.split("https://t.co/").str[0]
# Replace any occurence of @user with <tag>
regex_tag = r"(^|[^@\w])@(\w{1,15})\b"
data["tweet_text_clean"] = data["tweet_text_clean"].apply(
    lambda x: re.sub(regex_tag, "<tag>", x)
)

## Add the text of the image if it exists
# Number of files in the folder
n_files = len(os.listdir(IMG_TEXT_FOLDER))
# Names of the files
files = os.listdir(IMG_TEXT_FOLDER)
# Add new column in the dataset for the image text, filled with None
data["img_text"] = [None] * len(data)
# Load each file and add the text to the dataset to the corresponding index
for file in files:
    index = int(file.split(".")[0])

    # Open the file (json)
    with open(IMG_TEXT_FOLDER + file) as f:
        file_data = json.load(f)

        # Add the text to the dataset
        data.loc[data["index"] == index, "img_text"] = file_data["img_text"]
data["text_in_image"] = data["img_text"].isna().apply(lambda x: not x)


## Add the hate_speech label
# replace the labels with a single label hateful or not
data["hate_speech"] = data.apply(
    lambda x: np.mean([0 if i == 0 else 1 for i in x["labels"]]), axis=1
)
data["binary_hate"] = data["hate_speech"].apply(lambda x: 1 if x >= 0.5 else 0)

## Add the split
# Load the splitsb
train = pd.read_csv(SPLITS_FOLDER + "train_ids.txt", header=None)
test = pd.read_csv(SPLITS_FOLDER + "test_ids.txt", header=None)
val = pd.read_csv(SPLITS_FOLDER + "val_ids.txt", header=None)

# Add the split to the dataset if the index is in the split
data["split"] = "train"
data.loc[data["index"].isin(test[0]), "split"] = "test"
data.loc[data["index"].isin(val[0]), "split"] = "val"

## Save the dataset
data.to_csv(OUTPUT_CSV, index=False)
# Second version of the dataset with only tweets which have a text in the image
data2 = data[data["text_in_image"]]
data2 = data2.drop(columns=["text_in_image"])
# Save the dataset
data2.to_csv(OUTPUT_TEXT_IN_IMAGE, index=False)
