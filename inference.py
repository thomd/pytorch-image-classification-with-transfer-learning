import config
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from torch import nn
import torchmetrics
import cv2
import math
import numpy as np
import argparse
import pathlib
import torch
import os

def image_grid(tensor, true_labels, pred_labels, path, nrow=8, limit=None, pad=12):
    tensor = tensor.cpu()
    if limit is not None:
        tensor = tensor[:limit, ::]
        true_labels = true_labels[:limit]
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + pad), int(tensor.size(3) + pad)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + pad, width * xmaps + pad), 1)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            t = tensor[k]
            img = cv2.UMat(np.asarray(np.transpose(t.numpy(), (1, 2, 0)) * 255).astype('uint8'))
            text = f'{str(pred_labels[k])}'
            image = cv2.putText(img, text, (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, (255, 255, 255), 12, cv2.LINE_AA) # outline
            color = (0, 255, 0) if pred_labels[k] == true_labels[k] else (255, 0, 0)
            image = cv2.putText(img, text, (10, 70), cv2.FONT_HERSHEY_PLAIN, 5, color, 3, cv2.LINE_AA)
            t = transforms.ToTensor()(image.get())
            grid.narrow(1, y * height + pad, height - pad).narrow(2, x * width + pad, width - pad).copy_(t)
            k += 1

    filename = f'batch_{nmaps}.png'
    save_image(grid, os.path.join(path, filename))
    return filename


def inference(args):
    batch_size = args['batch']

    test_transforms = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    print('[INFO] loading the test dataset...')
    test_dataset = datasets.ImageFolder(root=args['dataset_path'], transform=test_transforms)
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

    # initialize metrics
    metric = torchmetrics.Accuracy()
    metric.to(config.DEVICE)

    # switch off autograd
    with torch.no_grad():
        for (batch_idx, (images, labels)) in enumerate(test_loader):
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            preds = model(images)

            pred_labels = preds.max(1).indices
            acc = metric(labels, pred_labels)

            # save images for first batch
            if batch_idx == 0:
                image_path = image_grid(images, np.asarray(labels.cpu()), np.array(pred_labels.cpu()), args['output_path'])
                print(f'[INFO] image location: {image_path}')

    if args['show_metrics']:
        acc = metric.compute()
        print(f"Accuracy:        {acc:.3f}")
        print(f'True Positives:  TODO')
        print(f'True Negatives:  TODO')
        print(f'False Positives: TODO')
        print(f'False Negatives: TODO')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference of Test Images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=pathlib.Path, required=True, help='path to trained model model')
    parser.add_argument('--dataset-path', type=pathlib.Path, default=os.path.join(config.DATASET_PATH, config.TEST), help='path to test dataset')
    parser.add_argument('--output-path', type=pathlib.Path, default='output', help='output path')
    parser.add_argument('--image-path', type=pathlib.Path, help='path to test images instead of batch')
    parser.add_argument('--show-metrics', type=bool, default=True, help='print inference metrics')
    parser.add_argument('--batch', type=int, default=config.PRED_BATCH_SIZE, help='batch size')
    args = vars(parser.parse_args())

    inference(args)
