# PyTorch Image Classification with Transfer Learning

In this experiment we train an image classifier using [transfer learning?](https://nbviewer.jupyter.org/github/thomd/pytorch-image-classification-with-transfer-learning/blob/main/transfer-learning.ipynb) of the pre-trained convolutional neural network **ResNet-50**.

For this example, we use the [Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset/) to train the model.

## Local Setup

    pyenv shell 3.9.10
    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

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

## Run in Google Colab

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
    !python build_dataset.py

    %load_ext tensorboard

    %tensorboard --logdir=./results

    !python train.py --help
    !python train.py --type fine-tuning

    !python inference.py --help
    !python inference.py --model /path/to/model.pth

    from IPython.display import Image
    display(Image('/path/to/iamge.png'))
