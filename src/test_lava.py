from src.dataset import CustomDataset
from src.utils import ROOT_DIR, load_config_model
from src.models import Lava
from torch.utils.data import DataLoader
import os
import torch
import time
from torchvision import transforms
import json
import sys

import click


@click.command()
@click.option(
    "-c",
    "--config_file",
    type=str,
    multiple=False,
    required=True,
    help="""name of the config file (add the extension to it's name)""",
)
def evalutate_lava(config_file: str):
    """Script to evalute the lava model"""
    # Script to test the lava model at least 60G
    batch_size = 2
    config = load_config_model(config_file)
    run_name = config["model_name"]
    print("Run name:", run_name)
    result_path = os.path.join(
        ROOT_DIR, "data", "results", f"{run_name}_prediction_on_test.json"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose(
        [transforms.Resize((299, 299)), transforms.ToTensor()]
    )
    # load the data
    dataset = CustomDataset(
        csv_file=os.path.join(
            ROOT_DIR, "data", "MMHS150K", "MMHS150K_text_in_image.csv"
        ),
        img_dir=os.path.join(ROOT_DIR, "data", "MMHS150K", "img_resized/"),
        split="test",
        transform=transform,
    )
    test_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sys.stdout.write(f'Loading the model_{time.strftime("%H:%M:%S", time.localtime())}')
    model = Lava(config_file, device)

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
