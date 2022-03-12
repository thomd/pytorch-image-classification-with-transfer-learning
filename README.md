# PyTorch Image Classification with Transfer Learning

Source: [PyTorch: Transfer Learning and Image Classification](https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/)

## Setup

    python -m venv .venv
    source .venv/bin/activate
    pip install kaggle tqdm imutils numpy torch torchvision matplotlib jupyterlab

## Dataset

    kaggle datasets list -s flowers
    kaggle datasets download -d imsparsh/flowers-dataset
    unzip flowers-dataset.zip -d flower-photos
    python build_dataset.py

## Train by Feature Extraction

    ulimit -n 10240
    python train_feature_extraction.py

## Train by Fine Tuning

    python fine_tune.py

## Inference

    ipython inference.py --model output/finetune_model.pth
    ipython inference.py --model output/feature_extraction_model.pth





