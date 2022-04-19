# PyTorch Image Classification with Transfer Learning

In this experiment we train an image classifier using [transfer learning](https://nbviewer.jupyter.org/github/thomd/pytorch-image-classification-with-transfer-learning/blob/main/transfer-learning.ipynb) of the pre-trained convolutional neural network **ResNet-50**.

## Local Setup

    conda env create -f environment.yml python=3.9
    conda activate ictl

### Image Data

Images must be structured like this:

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

For this example, we use the [Flowers Dataset](https://www.kaggle.com/datasets/imsparsh/flowers-dataset/) to train a flower-classification model:

    kaggle datasets list -s flowers
    kaggle datasets download -d imsparsh/flowers-dataset
    unzip flowers-dataset.zip -d flower-photos

For ease of demonstration, we only use the `train` part of the dataset and do a train-validate-test split ourself:

    python build_dataset.py --help
    python build_dataset.py --images-path flower-photos/train

### Train and Validate Model by Transfer-Learning

    python train.py --help
    python train.py --show-labels

    tensorboard --logdir=results
    open http://localhost:6006

Train by Feature Extraction

    python train.py --type feature-extraction --epochs 50 --batch 32 --lr 0.001 --export-onnx

Train by Fine Tuning

    python train.py --type fine-tuning --epochs 50 --batch 32 --lr 0.0005 --export-onnx

### Test Model

Test a specific model from all experiments within the `results` folder:

    python inference.py --model results/{best-experiment}/best_model.pth --batch 16

Print result and label of a class with

    open output/batch_16.png
    python train.py --show-labels

### Prediction of an individual Image

    python inference.py --model results/.../best_model.pth --image-path /path/to/image.jpg

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

    !python build_dataset.py --images-path flower-photos/train

    %load_ext tensorboard
    %tensorboard --logdir=./results

    !python train.py --show-labels
    !python train.py --epochs 60 --batch 64 --lr 0.0001

    !python inference.py --model results/.../best_model.pth --batch 16
    from IPython.display import Image
    display(Image('output/batch_16.png'))

## Run Inference Endpoint with Fast API

This endpoint expects a trained **ONNX classification model** `best_model.onnx` in the root folder:

    cp results/{best-experiment}/best_model.onnx .

Either start [uvicorn](https://www.uvicorn.org/) web server with

    uvicorn service:app
    curl -F "file=@image.jpg" -H "Content-Type: multipart/form-data" http://127.0.0.1:8000/image

or build and run as **Docker container** with

    docker-compose up -d --build
    docker logs image-classification
    curl -F "file=@image.jpg" -H "Content-Type: multipart/form-data" http://127.0.0.1:8000/image
    docker-compose down

## Deploy Service as Azure Container Instance

Use Azure Container Instances (ACI) to run serverless Docker containers in Azure.

Create Azure Container Registry (ACR):

    az login
    az group create -n <group> -l <location>
    az acr create -g <group> -n <reg> --sku Basic --admin-enabled true
    az acr list -g <group> -o table

Build & upload docker image to ACR

    docker build -t image-classification-api .
    docker tag image-classification-api <reg>.azurecr.io/image-classification-api:v1
    az acr login -n <reg>
    docker push <reg>.azurecr.io/image-classification-api:v1

Create container (find username and password here: [Azure Portal](https://portal.azure.com) > Container Registry > Access Keys)

    az container create -g <group> -n <container> --image <reg>.azurecr.io/image-classification-api:v1 --dns-name-label <label> --ports 80
    az container show -g <group> -n <container> --query "{FQDN:ipAddress.fqdn,ProvisioningState:provisioningState}"
    az container logs -g <group> -n <container> --follow

Open SwaggerUI
    
    open http://<label>.<location>.azurecontainer.io

Delete resource group
    
    az group delete -n <group>
