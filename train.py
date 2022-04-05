import config
from torchvision import models
from torchvision import transforms
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import re
import time
import argparse
import pathlib


def create_exeriment_path(base_path, *params):
    dirpath = pathlib.Path(base_path)
    os.makedirs(base_path, exist_ok=True)
    experiments = [0]
    for exp in dirpath.iterdir():
        if exp.is_dir():
            m = re.match(r'.*\/exp([0-9]+)', str(exp))
            if bool(m):
                experiments.append(int(m.groups()[0]))

    exp_n = np.max(experiments) + 1
    exp_path = f'{base_path}/exp{exp_n:03d}'
    for param in params:
        exp_path += f'_{str(param)}'
    os.makedirs(exp_path, exist_ok=True)
    return exp_path


def train(args):
    if args['batch'] != None:
        batch_size = args['batch']
    elif args['type'] == 'fine-tuning':
        batch_size = config.FINETUNE_BATCH_SIZE
    else:
        batch_size = config.FEATURE_EXTRACTION_BATCH_SIZE

    if args['lr'] != None:
        lr = args['lr']
    elif args['type'] == 'fine-tuning':
        lr = config.LR_FINETUNE
    else:
        lr = config.LR

    train_transforms = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        #A.VerticalFlip(p=0.5),
        A.Perspective(scale=(0.05, 0.05), fit_output=False, p=0.5),
        A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.5),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    train_image_folder = os.path.join(args['dataset_path'], config.TRAIN)
    train_dataset = datasets.ImageFolder(root=train_image_folder, transform=lambda img:train_transforms(image=np.array(img))['image'])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_image_folder = os.path.join(args['dataset_path'], config.VAL)
    val_dataset = datasets.ImageFolder(root=val_image_folder, transform=val_transforms)

    if args['show_labels']:
        print(f'{"NAME":<14} INDEX')
        for label in train_dataset.class_to_idx.items():
            name, idx = label
            print(f'{name:<14} {idx}')
        return

    num_classes = len(train_dataset.classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True if config.DEVICE == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True if config.DEVICE == 'cuda' else False)

    if args['show_images']:
        (batch, _) = next(iter(DataLoader(train_dataset, batch_size=args['show_images'], shuffle=True)))
        deNormalize = transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444])
        plt.figure(figsize=(20, 10))
        plt.imshow(transforms.ToPILImage()(make_grid(deNormalize(batch))))
        plt.axis('off')
        image_path = os.path.join(args['output_path'], 'train_images.jpg')
        print(f'[INFO] image location: {image_path}')
        plt.savefig(image_path)
        return

    # load model // TODO: add other nets
    if args['model'] == 'resnet':
        model = models.resnet50(pretrained=True)

    # FEATURE EXTRACTION
    if args['type'] == 'feature-extraction':
        # freeze parameters to non-trainable (by default they are trainable)
        for param in model.parameters():
            param.requires_grad = False

        # append a new classification top to our feature extractor and pop it on to the current device
        output_features = model.fc.in_features
        model.fc = nn.Linear(output_features, num_classes)
        model = model.to(config.DEVICE)

        # initialize loss function and optimizer (notice that we are only providing the parameters of the classification top to our optimizer)
        loss_fn = nn.CrossEntropyLoss()

        if args['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(model.fc.parameters(), lr=lr, momentum=0.9)

    else:
        # FINE TUNE
        num_features = model.fc.in_features

        # loop over the modules of the model and set the parameters of batch normalization modules as not trainable
        for module, param in zip(model.modules(), model.parameters()):
            if isinstance(module, nn.BatchNorm2d):
                param.requires_grad = False

        # define the network head and attach it to the model
        model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        model = model.to(config.DEVICE)

        # initialize loss function and optimizer (notice that we are only providing the parameters of the classification top to our optimizer)
        loss_fn = nn.CrossEntropyLoss()

        if args['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # calculate steps per epoch for training and validation set
    train_steps = len(train_dataset) // batch_size
    val_steps = len(val_dataset) // batch_size

    # initialize a dictionary to store training history
    log = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f'\n[INFO] model:         {args["model"]}')
    print(f'[INFO] type:          {args["type"]}')
    print(f'[INFO] optimizer:     {args["optimizer"]}')
    print(f'[INFO] batch size:    {batch_size}')
    print(f'[INFO] learning rate: {lr}')
    start_time = time.time()
    experiment_path = create_exeriment_path(args['results_path'], args['model'], args['type'], batch_size, lr)
    print(f'[INFO] experiment:    {experiment_path}')
    print('')

    if args['tensorboard']:
        writer = SummaryWriter(experiment_path)

    epochs = args['epochs']
    for epoch in range(epochs):

        # set the model in training mode
        model.train()

        # initialize the total training and validation loss
        total_train_loss = 0
        total_val_loss = 0

        # initialize the number of correct predictions in the training and validation step
        train_correct = 0
        val_correct = 0
        best_val_correct = 0

        # loop over the training set
        for (batch_idx, (X, y)) in enumerate(train_loader):

            # send the input to the device
            (X, y) = (X.to(config.DEVICE), y.to(config.DEVICE))

            # perform a forward pass and calculate the training loss
            pred = model(X)
            loss = loss_fn(pred, y)

            # calculate the gradients
            loss.backward()

            # check if we are updating the model parameters and if so update them, and zero out the previously accumulated gradients
            if (batch_idx + 2) % 2 == 0:
                optimizer.step()
                optimizer.zero_grad()

            # add the loss to the total training loss so far and calculate the number of correct predictions
            total_train_loss += loss
            train_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # switch off autograd
        with torch.no_grad():

            # set the model in evaluation mode
            model.eval()

            # loop over the validation set
            for (X, y) in val_loader:

                # send the input to the device
                (X, y) = (X.to(config.DEVICE), y.to(config.DEVICE))

                # make the predictions and calculate the validation loss
                pred = model(X)
                total_val_loss += loss_fn(pred, y)

                # calculate the number of correct predictions
                val_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        # calculate the average training and validation loss
        avg_train_loss = (total_train_loss / train_steps).cpu().detach().numpy().item()
        avg_val_loss = (total_val_loss / val_steps).cpu().detach().numpy().item()

        # calculate the training and validation accuracy
        train_correct = train_correct / len(train_dataset)
        val_correct = val_correct / len(val_dataset)

        # serialize the model to disk
        if val_correct > best_val_correct:
            best_val_correct = val_correct
            torch.save(model, os.path.join(experiment_path, 'best_model.pth'))
            if args['export_onnx']:
                dummy_input = torch.randn(1, 3, 224, 224)
                torch.onnx.export(mnist, dummy_input, 'best_model.onnx', verbose=False, input_names=['image'], output_names=['prediction'])

        # update our training history
        log['train_loss'].append(avg_train_loss)
        log['train_acc'].append(train_correct)
        log['val_loss'].append(avg_val_loss)
        log['val_acc'].append(val_correct)
        if args['tensorboard']:
            writer.add_scalars('Loss', {'Train': avg_train_loss, 'Validation': avg_val_loss}, epoch)
            writer.add_scalars('Accuracy', {'Train': train_correct, 'Validation': val_correct}, epoch)

        # print the model training and validation information
        print(f'EPOCH: {epoch + 1}/{epochs}')
        print(f'Train Loss: {avg_train_loss:.6f}, Train Accuracy: {train_correct:.4f}')
        print(f'Validation Loss: {avg_val_loss:.6f}, Validation Accuracy: {val_correct:.4f}')

    # display the total time needed to perform the training
    end_time = time.time()
    print(f'Total time to train the model: {(end_time - start_time):.2f}s')
    if args['tensorboard']:
        writer.flush()
        writer.close()

    # plot the training loss and accuracy
    if args['plot']:
        plt.style.use('ggplot')
        plt.figure()
        plt.plot(log['train_loss'], label='train loss')
        plt.plot(log['val_loss'], label='val loss')
        plt.plot(log['train_acc'], label='train acc')
        plt.plot(log['val_acc'], label='val acc')
        plt.xlabel('Epoch')
        plt.ylabel('Loss / Accuracy')
        plt.legend()
        plt.savefig(os.path.join(experiment_path, 'accuracy_loss.png'))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer Learning of a CNN.')
    parser.add_argument('--dataset-path', type=pathlib.Path, default=config.DATASET_PATH, metavar='PATH', help=f'path to dataset (default: {config.DATASET_PATH})')
    parser.add_argument('--type', default='fine-tuning', choices=['feature-extraction', 'fine-tuning'], help='type of transfer learning (default: fine-tuning)')
    parser.add_argument('--model', default='resnet', choices=['resnet'], help='pre-trained model')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], help='type of optimizer (default: adam)')
    parser.add_argument('--plot', default=False, action='store_false', help='create image for loss/accuracy (default: False)')
    parser.add_argument('--tensorboard', action='store_true', default=True, help='write Tensorboard logs (default: True)')
    parser.add_argument('--results-path', type=pathlib.Path, default='./results', metavar='PATH', help='path to models and logs (default: ./results)')
    parser.add_argument('--batch', type=int, help='batch size')
    parser.add_argument('--lr', type=float, help='learning rate')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help=f'number of epochs (default: {config.EPOCHS})')
    parser.add_argument('--show-labels', action='store_true', help='show lables and exit')
    parser.add_argument('--show-images', type=int, default=None, metavar='NUM', help='show samples of augmented training images and exit')
    parser.add_argument('--output-path', type=pathlib.Path, default='output', metavar='PATH', help='output path for images and plots')
    parser.add_argument('--export-onnx', action='store_true', default=True, help='export model as *.pth and *.onnx')
    args = vars(parser.parse_args())

    train(args)
