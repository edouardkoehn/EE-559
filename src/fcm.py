import os

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from src.utils import ROOT_DIR, load_config_model

# ----------------------------
# Starting from here: models from the paper (https://arxiv.org/pdf/1910.03814.pdf)
# Paper of the dataset


class LSTMClassifier(nn.Module):
    """Long Short Term Memory Classifier:
    This model uses LSTM to classify the text
    """

    def __init__(
        self, embedding_dim, hidden_dim, vocab_size, output_size, batch_size, device
    ):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.device = device

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim).to(device)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True).to(device)
        self.hidden2out = nn.Linear(hidden_dim, output_size).to(device)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (
            torch.zeros(1, self.batch_size, self.hidden_dim).to(self.device),
            torch.zeros(1, self.batch_size, self.hidden_dim).to(self.device),
        )

    def forward(self, sentences):
        # sentences is a BatchEncoding object, we need to extract the input_ids
        sentence = sentences["input_ids"]
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(
            embeds.view(self.batch_size, sentence.shape[1], -1), self.hidden
        )
        # out_space = self.hidden2out(lstm_out[-1])
        # out_scores = F.log_softmax(out_space, dim=1)
        # Return the hidden layer output
        return lstm_out[:, -1, :]


# Get ImagNet IncetionV3 model
class InceptionV3(nn.Module):
    """InceptionV3 model:
    This model uses the InceptionV3 model to extract features from the images
    """

    def __init__(self, freeze_model=False):
        super(InceptionV3, self).__init__()

        self.inception = models.inception_v3(pretrained=False)
        self.inception.fc = nn.Identity()  # Remove the last layer to get the features

        if freeze_model:
            for param in self.inception.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.inception(x)


class FCM(nn.Module):
    """Feature Concatenation Model:
    This model concatenates the features of the image and the text to generate the output

    The structure of the model is as follows:
    1. Image features are extracted using InceptionV3 model
    2. Text features (both from the tweet and the image) are extracted using LSTM model
    3. The features are concatenated and passed through 4 fully connected layers to generate the output
    """

    def __init__(
        self,
        device,
        vocab_size,
        batch_size,
        output_size=1,
        freeze_image_model=False,
        freeze_text_model=False,
    ):
        super(FCM, self).__init__()

        self.device = device

        self.image_model = InceptionV3(freeze_image_model).to(device)
        inception_out_dim = 2048

        """
        text_embedding_dim = 100
        text_hidden_dim = 150
        self.text_model_tweet = LSTMClassifier(
            text_embedding_dim,
            text_hidden_dim,
            vocab_size,
            output_size,
            batch_size,
            device,
        ).to(device)
        self.text_model_img = LSTMClassifier(
            text_embedding_dim,
            text_hidden_dim,
            vocab_size,
            output_size,
            batch_size,
            device,
        ).to(device)
        """

        if freeze_text_model:
            for param in self.text_model_tweet.parameters():
                param.requires_grad = False
            for param in self.text_model_img.parameters():
                param.requires_grad = False

        # self.fc1 = nn.Linear(inception_out_dim + 2 * text_hidden_dim, 1024).to(device)
        self.fc1 = nn.Linear(inception_out_dim, 1024).to(device)
        self.fc2 = nn.Linear(1024, 512).to(device)
        self.fc3 = nn.Linear(512, 256).to(device)
        self.fc4 = nn.Linear(256, output_size).to(device)

        self.initilize_weights()

    def initilize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, image, tweet_text, img_text):

        # Extract features from the image
        image_features = self.image_model(image.to(self.device))
        # Turn the InceptionOutput object to a tensor
        image_features = torch.tensor(image_features[0].detach().cpu().numpy()).to(
            self.device
        )

        # Extract features from the texts
        # tweet_text_features = self.text_model_tweet(tweet_text.to(self.device))
        # img_text_features = self.text_model_img(img_text.to(self.device))

        # Concatenate the features
        """
        combined_features = torch.cat(
            (image_features, tweet_text_features, img_text_features), 1
        )"""
        combined_features = image_features

        # Pass the features through the fully connected layers
        x = F.relu(self.fc1(combined_features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)

        return x
