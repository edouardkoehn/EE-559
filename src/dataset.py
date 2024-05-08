import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, csv_file, img_dir, split, transform=None):
        """ __init__ function for CustomDataset

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

            
        image_index = self.dataset['index'].values[idx]
        img_path = self.img_dir + str(image_index) + '.jpg'

        image = Image.open(img_path)

        # If image has 1 channel, convert to 3 channels
        if image.mode != "RGB":
            image = image.convert("RGB")

        # If image has 1 channel, convert to 3 channels
        if image.mode != "RGB":
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = self.dataset[self.dataset["index"] == image_index][
            "binary_hate"
        ].values[0]
        tweet_text = self.dataset[self.dataset["index"] == image_index][
            "tweet_text_clean"
        ].values[0]
        img_text = self.dataset[self.dataset["index"] == image_index][
            "img_text"
        ].values[0]

        sample = {
            "image": image,
            "label": label,
            "tweet_text": tweet_text,
            "img_text": img_text,
            "index": image_index,
        }

        return sample
