# PyTorch Image Classification with Transfer Learning

Source: [PyTorch: Transfer Learning and Image Classification](https://pyimagesearch.com/2021/10/11/pytorch-transfer-learning-and-image-classification/)

## Local Setup

    python -m venv .venv
    source .venv/bin/activate
    pip install kaggle tqdm imutils numpy torch torchvision matplotlib jupyterlab

### Dataset

    kaggle datasets list -s flowers
    kaggle datasets download -d imsparsh/flowers-dataset
    unzip flowers-dataset.zip -d flower-photos
    python build_dataset.py

### Train by Feature Extraction

Using this method we are able to utilize CNNs to recognize classes it was never trained on!

Remove the fc-layer head from the pre-trained network and replace it with a softmax classifier. This method is super simple as it allows to treat the pre-trained CNN as a feature extractor and then pass those features through a logistic-regression classifier.

    ulimit -n 10240
    python train_feature_extraction.py

### Train by Fine Tuning

    python fine_tune.py

### Inference

    ipython inference.py --model output/finetune_model.pth
    ipython inference.py --model output/feature_extraction_model.pth

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




