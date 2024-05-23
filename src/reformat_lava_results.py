from src.dataset import CustomDataset
from src.utils import ROOT_DIR
from src.models import Lava
from torch.utils.data import DataLoader
import os
import torch
import time
from torchvision import transforms
import json
import sys

# Change to the path ot your data (prediction_lava.json)
path_lava_image = ...
result_path_reformated = os.path.join(
    ROOT_DIR, "data", "results", "Lava0S_prediction_on_test_reformated.json"
)
result_path_final = os.path.join(
    ROOT_DIR, "data", "results", "Lava0S_prediction_on_test_final.json"
)

with open(path_lava_image, "r") as f:
    prediction_image = json.load(f)

# Parsed the results
result_parsed = {}
for key, value in prediction_image.items():
    result_parsed[key] = {}
    generation = value.split("\nASSISTANT: ")
    if generation[1][-1:] != "}":
        generation[1] = "".join([generation[1], "}"])
    if generation[1][-2:] != '"}':
        generation[1] = "".join([generation[1], '"}'])
    try:
        result_parsed[key] = json.loads(generation[1])
    except json.JSONDecodeError as e:
        print("JSON decoding error:", key, e)
    except Exception as e:
        print("An unexpected error occurred:", key, e)
        continue
# Saved the parsed results
with open(result_path_reformated, "w") as f:
    json.dump(result_parsed, f)

# Check the classification label
i = 0
for key, value in result_parsed.items():
    if (value["Classification"] == "hateful") | (
        value["Classification"] == "not hateful"
    ):
        i += 1
print("Amount of data point without clear label", len(result_parsed) - i)

# Convert to the final output
result_final = {}
for key, value in result_parsed.items():
    if (value["Classification"] == "hateful") | (
        value["Classification"] == "not hateful"
    ):
        result_final[key] = value["Classification"]

# Save the final reformated json.
with open(result_path_final, "w") as f:
    json.dump(result_final, f)
