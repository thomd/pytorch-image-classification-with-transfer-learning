# Transfer Learning

## Setup

    python -m venv .venv
    source .venv/bin/activate
    pip install kaggle imutils numpy torch

## Dataset

    kaggle datasets list -s flowers
    kaggle datasets download -d imsparsh/flowers-dataset
    unzip flowers-dataset.zip -d flower-photos
    python build_dataset.py
