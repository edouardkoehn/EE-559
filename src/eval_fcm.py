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


# Load model saved in PARENT_DIR/results/fcm.pth
def load_fcm():
    model = FCM()
    model.load_state_dict(torch.load(os.path.join(PARENT_DIR, "results", "fcm.pth")))
    model.eval()
    return model
