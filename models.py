import torch
import torch.nn as nn
import torchvision

# TODO Task 1c - Implement a SimpleBNConv
class SimpleBNConv(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 8, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(8),
            nn.Conv2d(8, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
        )
        self.to(device)

        # Transfer the model weights to the GPU

# TODO Task 1f - Create a model from a pre-trained model from the torchvision
#  model zoo.


# TODO Task 1f - Create your own models


# TODO Task 2c - Complete TextMLP
class TextMLP(nn.Module):
    def __init__(self, vocab_size, sentence_len, hidden_size):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Embedding(vocab_size, hidden_size//2),
            nn.Flatten(),
            # ....
        )


# TODO Task 2c - Create a model which uses distilbert-base-uncased
#                NOTE: You will need to include the relevant import statement.
# class DistilBertForClassification(nn.Module):
#   ....