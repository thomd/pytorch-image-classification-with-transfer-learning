import torch
import os

IMAGES_PATH = 'flower-photos/train'
DATASET_PATH = 'dataset'

TEST_SPLIT = 0.1
VAL_SPLIT = 0.1
TRAIN = 'train'
VAL = 'val'
TEST = 'test'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

FEATURE_EXTRACTION_BATCH_SIZE = 256
FINETUNE_BATCH_SIZE = 64
PRED_BATCH_SIZE = 10
EPOCHS = 100
LR = 0.001
LR_FINETUNE = 0.0005
