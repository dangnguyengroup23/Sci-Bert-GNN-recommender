import torch

SEED = 8
MODEL_NAME = "allenai/scibert_scivocab_uncased"
MAX_SAMPLES = 12000
BERT_BATCH_SIZE = 64
GRAD_ACCUM_STEPS = 3
EPOCHS_BERT = 43
EPOCHS_GNN = 250
BERT_LR = 5e-5
GNN_LR = 5e-4
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "models"
NUM_WORKERS = 0
GNN_DIM = 128
KNN_K = 6
CONT_WEIGHT = 0.0
ACCUM_STEPS = GRAD_ACCUM_STEPS

import os
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)