import torch
import os

DATA_PATH = "flower-photos/train"
BASE_PATH = "dataset"

VAL_SPLIT = 0.1
TRAIN = os.path.join(BASE_PATH, "train")
VAL = os.path.join(BASE_PATH, "val")

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FEATURE_EXTRACTION_BATCH_SIZE = 256
FINETUNE_BATCH_SIZE = 64
PRED_BATCH_SIZE = 4
EPOCHS = 30
LR = 0.001
LR_FINETUNE = 0.0005

FEATURE_EXTRACTION_PLOT = os.path.join("output", "feature_extraction.png")
FINETUNE_PLOT = os.path.join("output", "finetune.png")
FEATURE_EXTRACTION_MODEL = os.path.join("output", "feature_extraction_model.pth")
FINETUNE_MODEL = os.path.join("output", "finetune_model.pth")

