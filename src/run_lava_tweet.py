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

# Script to test the lava model at least 60G
batch_size = 2
run_name = "Lava0S_tweet"
print("Run name:", run_name)
config_name = "config_Lava0S_tweet"
result_path = os.path.join(
    ROOT_DIR, "data", "results", f"{run_name}_prediction_on_test.json"
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor()])
# load the data
dataset = CustomDataset(
    csv_file=os.path.join(ROOT_DIR, "data", "MMHS150K", "MMHS150K_text_in_image.csv"),
    img_dir=os.path.join(ROOT_DIR, "data", "MMHS150K", "img_resized/"),
    split="test",
    transform=transform,
)
test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

sys.stdout.write(f'Loading the model_{time.strftime("%H:%M:%S", time.localtime())}')
model = Lava(os.path.join(ROOT_DIR, "data", "config", f"{config_name}.json"), device)

# Compute the prediction
output = {}
index = 0
print(f'Computing the predictions_{time.strftime("%H:%M:%S", time.localtime())}')
for i, data_dict in enumerate(test_loader):
    data = data_dict
    data["image"] = data["image"].to(device)
    prediction = model(data)
    for i in range(data["image"].shape[0]):
        output[int(prediction["index"][i])] = prediction["generation"][i]

    if index % 10 == 0:
        print(
            f'Items_{int(index*batch_size)}_{time.strftime("%H:%M:%S", time.localtime())}'
        )
        with open(result_path, "w") as f:
            json.dump(output, f)
    index += 1

# Save the resutls
with open(result_path, "w") as f:
    json.dump(output, f)
print(f'Process completed_{time.strftime("%H:%M:%S", time.localtime())}')

# Reformat the results
with open(result_path, "r") as f:
    result = json.load(f)

# Parsed the output
result_parsed = {}
for key, value in result.items():
    result_parsed[key] = {}
    generation = value.split("\nASSISTANT: ")
    if generation[1][-2:] != '"}':
        generation[1] = "".join([generation[1], '"}'])
    result_parsed[key] = json.loads(generation[1])
result_path = os.path.join(
    ROOT_DIR, "data", "results", f"{run_name}_prediction_on_test_reformated.json"
)
with open(result_path, "w") as f:
    json.dump(result_parsed, f)

# Convert to the final output
result_final = {}
for key, value in result_parsed.items():
    result_final[key] = value["Classification"]
result_path = os.path.join(
    ROOT_DIR, "data", "results", f"{run_name}_prediction_on_test_final.json"
)
with open(result_path, "w") as f:
    json.dump(result_final, f)
