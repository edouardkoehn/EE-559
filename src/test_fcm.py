import os
import sys

# Get PARENT_DIR
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import CustomDataset
from src.fcm import FCM, test_model


def test_fcm():
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
        img_dir=os.path.join(PARENT_DIR, "data", "MMHS150K", "img"),
        split="test",
        transform=transform,
    )

    # Load the test dataloader, want to run the test on the full test dataset
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, drop_last=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Choose the tokenization function
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # Load the model
    # Create the model
    vocab_size = len(tokenizer)
    fcm = FCM(
        device,
        vocab_size,
        32,
        output_size=1,
        freeze_image_model=True,
        freeze_text_model=False,
    ).to(device)

    # JSON to save predictions
    json_path = os.path.join(PARENT_DIR, "results", "fcm_predictions.json")

    # Run the test
    test_model(
        model=fcm,
        test_loader=test_loader,
        tokenizer=tokenizer,
        device=device,
        savefile_path=json_path,
    )


if __name__ == "__main__":
    test_fcm()
