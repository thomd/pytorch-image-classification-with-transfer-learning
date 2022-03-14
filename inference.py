import config
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
import matplotlib.pyplot as plt
from torch import nn
import argparse
import pathlib
import torch
import os


def inference(args):
    batch_size = args['batch']

    # torch.multiprocessing.freeze_support()

    test_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.MEAN, std=config.STD)
    ])

    # calculate the inverse mean and standard deviation
    inv_mean = [-m/s for (m, s) in zip(config.MEAN, config.STD)]
    inv_std = [1/s for s in config.STD]

    # define our de-normalization transform
    denormalize = transforms.Normalize(mean=inv_mean, std=inv_std)

    print('[INFO] loading the dataset...')
    test_dataset = datasets.ImageFolder(root=config.VAL, transform=test_transforms)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True if config.DEVICE == "cuda" else False)

    # check if we have a GPU available, if so, define the map location accordingly
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = "cpu"

    # load the model
    print('[INFO] loading the model...')
    model = torch.load(args['model'], map_location=map_location)
    model.to(config.DEVICE)
    model.eval()

    # grab a batch of test data
    batch = next(iter(test_loader))
    (images, labels) = (batch[0], batch[1])

    fig = plt.figure('Results', figsize=(10, 10))

    # switch off autograd
    with torch.no_grad():

        # send the images to the device
        images = images.to(config.DEVICE)

        # make the predictions
        print('[INFO] performing inference...')
        preds = model(images)

        # loop over all the batch
        for i in range(0, batch_size):

            # initalize a subplot
            # ax = plt.subplot(1, batch_size, i + 1)
            ax = plt.subplot((i % 2) + 1, 4, i + 1)

            # grab the image, de-normalize it, scale the raw pixel intensities to the range [0, 255], and change the channel ordering from channels first tp channels last
            image = images[i]
            image = denormalize(image).cpu().numpy()
            image = (image * 255).astype('uint8')
            image = image.transpose((1, 2, 0))

            # grab the ground truth label
            idx = labels[i].cpu().numpy()
            gtLabel = test_dataset.classes[idx]

            # grab the predicted label
            pred = preds[i].argmax().cpu().numpy()
            predLabel = test_dataset.classes[pred]

            # add the results and image to the plot
            info = 'Ground Truth: {}\nPredicted: {}'.format(gtLabel, predLabel)
            plt.imshow(image)
            plt.title(info)
            plt.axis('off')

        # show the plot
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference of Test Images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=pathlib.Path, required=True, help='path to trained model model')
    parser.add_argument('--batch', type=int, default=config.PRED_BATCH_SIZE, help='batch size')
    args = vars(parser.parse_args())

    inference(args)
