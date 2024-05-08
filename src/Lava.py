import json
import os
import time

from src.dataset import CustomDataset
from src.models import Lava
from src.utils import ROOT_DIR

# Load the dataset
dataset = CustomDataset(
    csv_file=os.path.join(ROOT_DIR, "data", "MMHS150K", "MMHS150K_text_in_image.csv"),
    img_dir=os.path.join(ROOT_DIR, "data", "MMHS150K", "img_resized/"),
    split="test",
)
print(time.strftime("%H:%M:%S", time.localtime()))
model = Lava(os.path.join(ROOT_DIR, "data", "config", "config_Lava0S.json"))
print(time.strftime("%H:%M:%S", time.localtime()))

output = {}

for i in range(0, 3):
    output[str(dataset[i]["index"])] = model(dataset[i])["generation"]

print(time.strftime("%H:%M:%S", time.localtime()))
print(output)

with open("results.json", "w") as f:
    json.dump(output, f)
