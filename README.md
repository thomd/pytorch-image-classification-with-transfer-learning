# PyTorch Image Classification with Transfer Learning

Source: [PyTorch: Transfer Learning and Image Classification](https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/)

## Local Setup

    python -m venv .venv
    source .venv/bin/activate
    pip install kaggle imutils numpy torch torchvision matplotlib jupyterlab

### Dataset

    kaggle datasets list -s flowers
    kaggle datasets download -d imsparsh/flowers-dataset
    unzip flowers-dataset.zip -d flower-photos
    python build_dataset.py

### Train by Feature Extraction

    ulimit -n 10240
    python train.py --type feature-extraction

### Train by Fine Tuning

    python train.py --type fine-tuning

### Inference

    python inference.py --model output/finetune_model.pth
    python inference.py --model output/feature_extraction_model.pth

## Run in Google Colab

Create new Colab Notebook and run these commands:

    %cd /content
    !nvidia-smi
    !pip install -q kaggle
    from google.colab import files
    uploaded = files.upload()
    !mkdir ~/.kaggle
    !mv kaggle.json ~/.kaggle/
    !chmod 600 /root/.kaggle/kaggle.json

    !git clone https://github.com/thomd/pytorch-image-classification-with-transfer-learning.git
    %cd pytorch-image-classification-with-transfer-learning
    !kaggle datasets download -d imsparsh/flowers-dataset
    !unzip -qq flowers-dataset.zip -d flower-photos
    !python build_dataset.py
    %load_ext tensorboard

    %tensorboard --logdir=./runs

    !python train.py --help

