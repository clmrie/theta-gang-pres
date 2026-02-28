import os
import numpy as np
import torch

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

# Device
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

# Paths
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dataset')
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'outputs')
FIGURES_DIR = os.path.join(OUTPUT_DIR, 'figures')
PARQUET_NAME = "M1199_PAG_stride4_win108_test.parquet"
JSON_NAME = "M1199_PAG.json"

# Sequence
MAX_SEQ_LEN = 128

# Architecture
EMBED_DIM = 64
NHEAD = 4
NUM_LAYERS = 2
DROPOUT = 0.2

# Data augmentation
SPIKE_DROPOUT = 0.15
NOISE_STD = 0.5

# Training
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 7
BATCH_SIZE = 64
N_FOLDS = 5

# Loss weights
LAMBDA_D = 1.0
LAMBDA_FEAS = 10.0
