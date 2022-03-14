import config
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
import pathlib

def train(args):
    if args['batch'] != None:
        batch_size = args['batch']
    elif args['type'] == 'fine-tuning':
        batch_size = config.FINETUNE_BATCH_SIZE
    else:
        batch_size = config.FEATURE_EXTRACTION_BATCH_SIZE

    # torch.multiprocessing.set_sharing_strategy('file_system')
    # torch.multiprocessing.freeze_support()

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    train_dataset = datasets.ImageFolder(root=config.TRAIN, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True if config.DEVICE == "cuda" else False)

    val_dataset = datasets.ImageFolder(root=config.VAL, transform=val_transforms)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count(), pin_memory=True if config.DEVICE == "cuda" else False)

    # load ResNet50 model as feature extractor
    model = models.resnet50(pretrained=True)

    if args['type'] == 'feature-extraction':
        # freeze parameters to non-trainable (by default they are trainable)
        for param in model.parameters():
            param.requires_grad = False

        # append a new classification top to our feature extractor and pop it on to the current device
        output_features = model.fc.in_features
        model.fc = nn.Linear(output_features, len(train_dataset.classes))
        model = model.to(config.DEVICE)

        # initialize loss function and optimizer (notice that we are only providing the parameters of the classification top to our optimizer)
        loss_fn = nn.CrossEntropyLoss()

        if args['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.fc.parameters(), lr=config.LR)
        else:
            optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.001, momentum=0.9)

    else:
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
            nn.Linear(256, len(train_dataset.classes))
        )
        model = model.to(config.DEVICE)

        # initialize loss function and optimizer (notice that we are only providing the parameters of the classification top to our optimizer)
        loss_fn = nn.CrossEntropyLoss()

        if args['optimizer'] == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
        else:
            optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # calculate steps per epoch for training and validation set
    train_steps = len(train_dataset) // batch_size
    val_steps = len(val_dataset) // batch_size

    # initialize a dictionary to store training history
    log = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    print(f'[INFO] type:       {args["type"]}')
    print(f'[INFO] optimizer:  {args["optimizer"]}')
    print(f'[INFO] batch size: {batch_size}')
    start_time = time.time()
    if args['tensorboard']:
        writer = SummaryWriter(args['log_path'])

    epochs = args['epochs']
    for epoch in tqdm(range(epochs)):

        # set the model in training mode
        model.train()

        # initialize the total training and validation loss
        total_train_loss = 0
        total_val_loss = 0

        # initialize the number of correct predictions in the training and validation step
        train_correct = 0
        val_correct = 0

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
        plt.savefig(config.FEATURE_EXTRACTION_PLOT)

    # serialize the model to disk
    torch.save(model, config.FEATURE_EXTRACTION_MODEL)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transfer Learning of a CNN.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--type', default='fine-tuning', choices=['feature-extraction', 'fine-tuning'], help='type of transfer learning')
    parser.add_argument('--optimizer', default='adam', choices=['adam', 'sgd'], help='type of optimizer')
    parser.add_argument('--plot', default=False, type=bool, help='create image for loss/accuracy')
    parser.add_argument('--tensorboard', default=True, action=argparse.BooleanOptionalAction, help='write Tensorboard logs')
    parser.add_argument('--log-path', type=pathlib.Path, default='./runs', help='path to Tensorboard logs')
    parser.add_argument('--batch', type=int, help='batch size')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS, help='number of epochs')
    args = vars(parser.parse_args())

    # train(args)
    print(args)
