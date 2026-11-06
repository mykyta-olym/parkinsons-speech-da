"""CNN modely nad log-mel spektrogramami PC-GITA."""

from __future__ import annotations

import torch
import torch.nn as nn

from src.config import NUM_CLASSES
from src.models.domain_adaptation import GradientReversalLayer


class SpectrogramFeatureExtractor(nn.Module):
    """Kompaktný 2D CNN extraktor pre log-mel spektrogram."""

    def __init__(self, feature_dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, feature_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(x))


class SpectrogramCNN(nn.Module):
    """Source-only CNN klasifikátor nad spektrogramom."""

    def __init__(self, feature_dim: int = 128, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.feature_extractor = SpectrogramFeatureExtractor(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        return logits, features


class SpectrogramDANN(nn.Module):
    """DANN nad spektrogramovým CNN extraktorom."""

    def __init__(
        self,
        feature_dim: int = 128,
        num_classes: int = NUM_CLASSES,
        num_domains: int = 2,
        lambda_val: float = 1.0,
    ):
        super().__init__()
        self.feature_extractor = SpectrogramFeatureExtractor(feature_dim=feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.grl = GradientReversalLayer(lambda_val=lambda_val)
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, num_domains),
        )

    def forward(self, x: torch.Tensor):
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        domain_logits = self.domain_classifier(self.grl(features))
        return logits, domain_logits, features
