import json
import os
import time
from src.dataset import CustomDataset
from src.models import Flamingo0S
from src.utils import ROOT_DIR
print(time.time())

dataset = CustomDataset(
    csv_file=os.path.join(ROOT_DIR, "data", "MMHS150K", "MMHS150K_text_in_image.csv"),
    img_dir=os.path.join(ROOT_DIR, "data", "MMHS150K", "img_resized/"),
    split="test",
)

model = Flamingo0S(
    config_path=os.path.join(ROOT_DIR, "data", "config", "config_Flamingo0S.json")
)
print(model)

res_dict = {}

for i in range(10):
    prediction = model(dataset[i])
    res_dict[f"{i}"] = prediction

res_path = os.path.join(ROOT_DIR, "data_test.json")
with open(res_path, "w") as f:
    json.dump(res_dict, f)

print(time.time())