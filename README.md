# PyTorch Image Classification with Transfer Learning

In this experiment we train an image classifier using [transfer learning?](https://nbviewer.jupyter.org/github/thomd/pytorch-image-classification-with-transfer-learning/blob/main/transfer-learning.ipynb) of the pre-trained convolutional neural network **ResNet-50**.

For this example, we use the [Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset/) to train the model.

## Local Setup

    conda env create -f environment.yml python=3.9
    conda activate ictl

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

Each class has it's own directory for the images. The images are then labeled with the class taken from the directory name.

### Create Dataset

    kaggle datasets list -s flowers
    kaggle datasets download -d imsparsh/flowers-dataset
    unzip flowers-dataset.zip -d flower-photos
    python build_dataset.py --help
    python build_dataset.py --images-path flower-photos/train

### Train by Feature Extraction

    ulimit -n 10240
    python train.py --help
    python train.py --show-labels
    python train.py --type feature-extraction

### Train by Fine Tuning

    python train.py --type fine-tuning

### Inference

    python inference.py --model output/finetune_model.pth
    python inference.py --model output/feature_extraction_model.pth

## Train in Google Colab

Create new [Colab Notebook](https://colab.research.google.com) and run these commands:

    %cd /content
    !nvidia-smi
    !pip install -q kaggle torchmetrics albumentations==1.1.0 opencv-python-headless==4.2.0.34
    from google.colab import files
    uploaded = files.upload()
    !mkdir ~/.kaggle
    !mv kaggle.json ~/.kaggle/
    !chmod 600 /root/.kaggle/kaggle.json

    !git clone https://github.com/thomd/pytorch-image-classification-with-transfer-learning.git
    %cd pytorch-image-classification-with-transfer-learning
    !kaggle datasets download -d imsparsh/flowers-dataset
    !unzip -qq flowers-dataset.zip -d flower-photos

    !python build_dataset.py --help
    !python build_dataset.py --show-tree

    %load_ext tensorboard

    %tensorboard --logdir=./results

    !python train.py --help
    !python train.py --show-labels
    !python train.py --epochs 60 --batch 64 --lr 0.0001 --export-onnx

    !python inference.py --help
    !python inference.py --model results/best_model.pth --batch 16

    from IPython.display import Image
    display(Image('/path/to/image.png'))

## Inference Endpoint with Fast API

This endpoint expects a trained **ONNX classification model** `best_model.onnx` in the root folder.

Convert `best_model.pth` to `best_model.onnx` with:

    import torch
    model = torch.load('best_model.pth', map_location='cpu')
    torch.onnx.export(model, torch.randn(1, 3, 224, 224), 'best_model.onnx')

Either start [uvicorn](https://www.uvicorn.org/) web server with

    uvicorn service:app --reload
    curl -F "file=@image.jpg" -H "Content-Type: multipart/form-data" http://127.0.0.1:8000/image

or run as **Docker container** with

    docker-compose up -d --build
    curl -F "file=@image.jpg" -H "Content-Type: multipart/form-data" http://127.0.0.1:8000/image
    docker-compose down
