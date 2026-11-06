"""Zakladna konfiguracia projektu."""

import os

import torch

# Zariadenie
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Cesty
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, 'data')

# Trening
BATCH_SIZE = 32
BATCH_SIZE_DA = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
DA_EPOCHS = 100
DA_LR = 0.0005
RANDOM_SEED = 42

# Velkosti vrstiev
HIDDEN_SIZE = 128
FEATURE_DIM = 64

# Regularizacia
DROPOUT_RATE = 0.3
GRAD_CLIP_NORM = 1.0
WEIGHT_DECAY = 1e-4

# Domain adaptation
DANN_LAMBDA = 1.0
MMD_LAMBDA = 1.0
CORAL_LAMBDA = 30.0
CDAN_LAMBDA = 0.5
CONTRASTIVE_LAMBDA = 0.5
CONTRASTIVE_TEMP = 0.07
CONTRASTIVE_CONF = 0.75

# Statistika
N_BOOTSTRAP = 1000
SEEDS = [42, 7, 123, 2024, 99, 31, 256, 777, 1337, 0]

# Vseobecne rozmery
NUM_CLASSES = 2
NUM_PCGITA_DOMAINS = 7

# Legacy UCI vetva pouziva tieto rozmery. V hlavnej PC-GITA vetve sa pocet
# priznakov cita priamo z cache.
NUM_FEATURES = 12
NUM_FEATURES_FULL = 47
NUM_DOMAINS = 4
