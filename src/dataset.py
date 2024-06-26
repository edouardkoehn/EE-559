import pandas as pd
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CustomDataset(Dataset):
    """Custom dataset for the MMHS150 dataset"""

    def __init__(self, csv_file, img_dir, split, transform=None):
        """__init__ function for CustomDataset

        Args:
            csv_file (str): Path to the csv file containing the dataset information
            img_dir (str): Path to the directory containing the images
            split (str): 'train' or 'test' or 'val'
            transform (callable, optional): Optional transform to be applied on a sample
        """
        full_data = pd.read_csv(csv_file)
        self.dataset = full_data[full_data["split"] == split]
        self.img_dir = img_dir
        self.split = split
        self.transform = transform

        # Number of images in the dataset
        self.dataset_len = len(self.dataset)

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print('input_id',idx)
        
        #print('Marin',self.dataset["index"].values[idx].tolist())
        #print('edi',self.dataset[self.dataset.index == idx].index.to_list()[0])
        index=self.dataset["index"].values[idx].tolist()
       
        #index = self.dataset[self.dataset.index == idx].index.to_list()[0]
        
        #index=self.dataset["index"].values[idx]
        image_index=index
        #image_index = self.dataset["index"][index]
        
        img_path = self.img_dir + str(image_index) + ".jpg"
        image = Image.open(img_path)

        # If image has 1 channel, convert to 3 channels
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform != None:
            image = self.transform(image)
        else:
            # Turn image to tensor
            image = transforms.ToTensor()(image)

        label = int(
            self.dataset[self.dataset["index"] == image_index]["binary_hate"].values[0]
        )
        tweet_text = str(
            self.dataset[self.dataset["index"] == image_index][
                "tweet_text_clean"
            ].values[0]
        )
        img_text = str(
            self.dataset[self.dataset["index"] == image_index]["img_text"].values[0]
        )
        hate_confidence = self.dataset[self.dataset["index"] == image_index][
            "hate_speech"
        ].values[0]

        sample = {
            "image": image,
            "label": label,
            "tweet_text": tweet_text,
            "img_text": img_text,
            "index": image_index,
            "hate_confidence": hate_confidence,
        }

        return sample

    def get_data_from_index(self, idx):
        """method to extract one element of the dataset based on it's unique hashcode"""
        return self.dataset.query(f"index == {idx}")
