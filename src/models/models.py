"""
Týždeň 3-5: Modely klasifikátorov.

MLP (Cvičenie 4 - FF.ipynb) - feed-forward sieť na akustických príznakoch
CNN (Cvičenie 5 - CNN.ipynb) - konvolučná sieť (1D namiesto 2D, lebo naše dáta sú 1D)

Vzor architektúry: Cvičenie 3 - nn.Module, __init__, forward
"""

import torch.nn as nn
import torch.nn.functional as F

from src.config import DROPOUT_RATE, HIDDEN_SIZE, NUM_CLASSES, NUM_FEATURES

# ========================================================================
# MLP klasifikátor (Týždeň 4)
# Vzor: Cvičenie 4 - FF.ipynb (feed-forward sieť na MNIST)
# Rozdiel: namiesto 784 pixelov máme 12 akustických príznakov
# ========================================================================

class MLP(nn.Module):
    """
    Multi-Layer Perceptron pre klasifikáciu PD vs Healthy.
    Architektúra: vstup → skrytá → skrytá → výstup
    Vzor z Cvičenie 4: nn.Linear + aktivačné funkcie
    """

    def __init__(self, input_size=NUM_FEATURES, hidden_size=HIDDEN_SIZE,
                 output_size=NUM_CLASSES, dropout=DROPOUT_RATE):
        super().__init__()
        # Vzor z Cvičenie 3: definícia vrstiev v __init__
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc_output = nn.Linear(hidden_size // 2, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Vzor z Cvičenie 3-4: forward pass s ReLU aktiváciou
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc_output(x)
        return x


# ========================================================================
# CNN klasifikátor (Týždeň 5)
# Vzor: Cvičenie 5 - CNN.ipynb (Conv2d na MNIST/CIFAR)
# Rozdiel: používame Conv1d, lebo naše dáta sú 1D vektor príznakov
# https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
# ========================================================================

class CNN1D(nn.Module):
    """
    1D konvolučná sieť pre akustické príznaky.
    Rovnaký princíp ako Conv2d z cvičenia 5, ale pre 1D dáta.
    Input: (batch, num_features) → unsqueeze → (batch, 1, num_features)
    """

    def __init__(self, input_size=NUM_FEATURES, num_filters=16,
                 output_size=NUM_CLASSES, dropout=DROPOUT_RATE):
        super().__init__()

        # Konvolučné bloky - vzor z Cvičenie 5: nn.Sequential
        # Conv1d namiesto Conv2d, MaxPool1d namiesto MaxPool2d
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, num_filters, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(num_filters, num_filters * 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        self.dropout = nn.Dropout(dropout)

        # Po dvoch MaxPool(2): dĺžka = input_size // 4
        # Vzor z Cvičenie 5: x.view(x.size(0), -1) pre flatten
        flat_size = (num_filters * 2) * (input_size // 4)
        self.fc_output = nn.Linear(flat_size, output_size)

    def forward(self, x):
        # Pridáme kanálovú dimenziu: (batch, features) → (batch, 1, features)
        x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.conv2(x)
        # Flatten - vzor z Cvičenie 5
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc_output(x)
        return x
