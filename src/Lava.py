import json
import os

from src.dataset import CustomDataset
from src.models import Lava
from src.utils import ROOT_DIR

# Load the dataset
dataset = CustomDataset(
    csv_file=os.path.join(ROOT_DIR, "data", "MMHS150K", "MMHS150K_text_in_image.csv"),
    img_dir=os.path.join(ROOT_DIR, "data", "MMHS150K", "img_resized/"),
    split="test",
)

model = Lava(os.path.join(ROOT_DIR, "data", "config", "config_Lava0S.json"))

output = {}
for i in range(0, 2):
    print(model(dataset[i]))
    output[dataset[i]["index"]] = model(dataset[i])
    print(output)

with open("results.json", "wd") as f:
    json.dump(output, f)
