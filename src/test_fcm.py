import os
import sys
import json

# Get PARENT_DIR
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import CustomDataset
from src.fcm import FCM, test_model, acc, f1


def test_fcm(time_saved):
    """Test the FCM model on the MMHS150K dataset.

    Args:
        time_saved (str): The time the model was saved. To reload it.

    """

    # Load the normalization parameters
    means_std_path = os.path.join(PARENT_DIR, "data", "MMHS150K", "means_stds.csv")
    means_stds = pd.read_csv(means_std_path)

    # Minimal transformation for the images
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[
                    means_stds["mean_red"][0],
                    means_stds["mean_green"][0],
                    means_stds["mean_blue"][0],
                ],
                std=[
                    means_stds["std_red"][0],
                    means_stds["std_green"][0],
                    means_stds["std_blue"][0],
                ],
            ),
        ]
    )

    test_dataset = CustomDataset(
        csv_file=os.path.join(
            PARENT_DIR, "data", "MMHS150K", "MMHS150K_with_img_text.csv"
        ),
        img_dir=os.path.join(PARENT_DIR, "data", "MMHS150K", "img_resized/"),
        split="test",
        transform=transform,
    )

    # Load the test dataloader, want to run the test on the full test dataset
    batch_size = 32
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Choose the tokenization function
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    weight_path = os.path.join(PARENT_DIR, "results", "fcm_" + time_saved + ".pth")

    vocab_size = len(tokenizer)

    # Load the model
    fcm = FCM(
        device,
        vocab_size,
        batch_size,
        output_size=1,
        freeze_image_model=True,
        freeze_text_model=False,
    ).to(device)
    fcm.load_state_dict(torch.load(weight_path))

    # JSON to save predictions
    json_path = os.path.join(
        PARENT_DIR, "results", "fcm_predictions_" + time_saved + ".json"
    )

    # Run the test
    test_model(
        model=fcm,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        savefile_path=json_path,
    )


if __name__ == "__main__":
    time_saved = "110524_2134"
    test_fcm(time_saved=time_saved)

    # Load dataset
    DATASET_PATH = os.path.join(
        PARENT_DIR, "data", "MMHS150K", "MMHS150K_with_img_text.csv"
    )
    df = pd.read_csv(DATASET_PATH)

    RESULTS_PATH = os.path.join(
        PARENT_DIR, "results", "fcm_predictions_" + time_saved + ".json"
    )

    # Load predictions
    with open(RESULTS_PATH, "r") as f:
        predictions = json.load(f)

    # Get the indices on which predictions were made
    indices = [int(k) for k in predictions.keys()]

    # Get the corresponding data
    data = df[df["index"].isin(indices)]
    data["prediction"] = [predictions[str(i)] for i in data["index"]]

    # Compute metrics
    acc_score = acc(data["prediction"], data["label"])
    f1_score = f1(data["prediction"], data["label"])

    print(f"Accuracy: {acc_score}")
    print(f"F1: {f1_score}")
