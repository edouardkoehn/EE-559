import os
import sys

# Get PARENT_DIR
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PARENT_DIR)

import time

import matplotlib.pyplot as plt
import pandas as pd
import requests
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms

from src.dataset import CustomDataset
from src.fcm import FCM, acc, eval_epoch, f1, train_epoch


# Some functions for the statistics
def update_metrics_log(metrics_names, metrics_log, new_metrics_dict):
    for i in range(len(metrics_names)):
        curr_metric_name = metrics_names[i]
        metrics_log[i].append(new_metrics_dict[curr_metric_name])
    return metrics_log


def save_metrics_log(metrics_names, metrics_log, path):
    metrics_log_df = pd.DataFrame(metrics_log, index=metrics_names).T
    metrics_log_df.to_csv(path, index=False)


# Main function
def run_fcm():
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

    train_dataset = CustomDataset(
        csv_file=os.path.join(
            PARENT_DIR, "data", "MMHS150K", "MMHS150K_with_img_text.csv"
        ),
        img_dir=os.path.join(PARENT_DIR, "data", "MMHS150K", "img_resized/"),
        split="train",
        transform=transform,
    )

    eval_dataset = CustomDataset(
        csv_file=os.path.join(
            PARENT_DIR, "data", "MMHS150K", "MMHS150K_with_img_text.csv"
        ),
        img_dir=os.path.join(PARENT_DIR, "data", "MMHS150K", "img_resized/"),
        split="val",
        transform=transform,
    )

    batch_size = 8
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # Choose the tokenization function
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

    # Create the model
    vocab_size = len(tokenizer)
    fcm = FCM(
        device,
        vocab_size,
        batch_size,
        output_size=1,
        freeze_image_model=True,
        freeze_text_model=False,
    ).to(device)

    # Choose the optimizer and the loss function
    optimizer = torch.optim.Adam(fcm.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()

    # Choose the metrics
    metrics = {"ACC": acc, "F1-weighted": f1}

    n_epochs = 5

    # Train the model
    train_loss_log, test_loss_log = [], []
    metrics_names = list(metrics.keys())
    train_metrics_log = [[] for i in range(len(metrics))]
    test_metrics_log = [[] for i in range(len(metrics))]

    for epoch in range(n_epochs):
        train_loss, train_metrics = train_epoch(
            fcm, optimizer, criterion, metrics, train_loader, tokenizer, device
        )

        test_loss, test_metrics = eval_epoch(
            fcm, criterion, metrics, eval_loader, tokenizer, device
        )

        train_loss_log.append(train_loss)
        train_metrics_log = update_metrics_log(
            metrics_names, train_metrics_log, train_metrics
        )

        test_loss_log.append(test_loss)
        test_metrics_log = update_metrics_log(
            metrics_names, test_metrics_log, test_metrics
        )

    # Save the metrics in PARENT_DIR/results/fcm_train_metrics_DDMMYY_HHMM.csv
    train_results_path = os.path.join(
        PARENT_DIR,
        "results",
        "fcm_train_results" + time.strftime("%d%m%y_%H%M") + ".csv",
    )
    test_results_path = os.path.join(
        PARENT_DIR,
        "results",
        "fcm_test_results" + time.strftime("%d%m%y_%H%M") + ".csv",
    )
    save_metrics_log(metrics_names, train_metrics_log, train_results_path)
    save_metrics_log(metrics_names, test_metrics_log, test_results_path)

    # Save the model
    torch.save(fcm.state_dict(), os.path.join(PARENT_DIR, "results", "fcm.pth"))


# Run the function
if __name__ == "__main__":
    run_fcm()
