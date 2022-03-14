# PyTorch Image Classification with Transfer Learning

Source: [PyTorch: Transfer Learning and Image Classification](https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/)

## Local Setup

    python -m venv .venv
    source .venv/bin/activate
    pip install kaggle imutils numpy torch torchvision tensorboard matplotlib jupyterlab

### Image Data

Image folder expects the files and directories to be constructed like this:

    .
    └── images
        ├── label_1
        │   ├── image_0.jpg
        │   ├── image_1.jpg
        │   └── image_2.jpg
        └── label_2
            ├── image_3.jpg
            └── image_4.jpg

Each class has it's own directory (cat and dog) for the images. The images are then labeled with the class taken from the directory name.

### Create Dataset

    kaggle datasets list -s flowers
    kaggle datasets download -d imsparsh/flowers-dataset
    unzip flowers-dataset.zip -d flower-photos
    python build_dataset.py --help
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

Create new [Colab Notebook](https://colab.research.google.com) and run these commands:

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
    !python train.py --type fine-tuning

    !python inference.py --help
    !python inference.py --model output/finetune_model.pth
    from IPython.display import Image
    display(Image('output/inference.png'))
