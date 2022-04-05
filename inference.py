import config
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.utils import save_image
from torch import nn
import torchmetrics
import cv2
from PIL import Image
import math
import numpy as np
import argparse
import pathlib
import torch
import os
import requests

def image_grid(tensor, true_labels, pred_labels, path, nrow=8, limit=None, pad=12):
    deNormalize = transforms.Normalize(mean=[-2.118, -2.036, -1.804], std=[4.367, 4.464, 4.444])
    tensor = deNormalize(tensor).cpu()
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

    inference_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # check if we have a GPU available, if so, define the map location accordingly
    if torch.cuda.is_available():
        map_location = lambda storage, loc: storage.cuda()
    else:
        map_location = 'cpu'

    # load the model
    print('[INFO] loading the model...')
    model = torch.load(args['model'], map_location=map_location)
    model.to(config.DEVICE)
    model.eval()

    if args['image_path']:
        image_path = pathlib.Path(args['image_path'])
        if image_path.exists():
            image = Image.open(str(image_path)).convert('RGB')
            image = inference_transforms(image)
            image = image.to(config.DEVICE)
            image = image.unsqueeze(0)
            preds = model(image)
            pred_labels = preds.max(1).indices.cpu()
            print(f'[INFO] Image {image_path} is of class: {pred_labels.item()}')
        else:
            print(f'[ERROR] Image {image_path} does not exist')

    if args['image_url']:
        url = args['image_url']
        image = Image.open(requests.get(url, stream=True).raw)
        image = inference_transforms(image)
        image = image.to(config.DEVICE)
        image = image.unsqueeze(0)
        preds = model(image)
        pred_labels = preds.max(1).indices.cpu()
        print(f'[INFO] Image {url} is of class: {pred_labels.item()}')

    else:
        print('[INFO] loading the test dataset ...')
        test_image_folder = os.path.join(args['dataset_path'], config.TEST)
        test_dataset = datasets.ImageFolder(root=test_image_folder, transform=inference_transforms)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count(), pin_memory=True if config.DEVICE == 'cuda' else False)

        # initialize metrics
        metric_acc = torchmetrics.Accuracy()
        metric_acc.to(config.DEVICE)
        metric_confmat = torchmetrics.ConfusionMatrix(num_classes=2)
        metric_confmat.to(config.DEVICE)
        metric_precision = torchmetrics.Precision(average='none', num_classes=2)
        metric_precision.to(config.DEVICE)
        metric_recall = torchmetrics.Recall(average='none', num_classes=2)
        metric_recall.to(config.DEVICE)

        # switch off autograd
        with torch.no_grad():
            for (batch_idx, (images, labels)) in enumerate(test_loader):
                (images, labels) = (images.to(config.DEVICE), labels.to(config.DEVICE))
                preds = model(images)

                pred_labels = preds.max(1).indices
                acc = metric_acc(labels, pred_labels)
                confmat = metric_confmat(labels, pred_labels)
                precision = metric_precision(labels, pred_labels)
                recall = metric_recall(labels, pred_labels)

                # save images for first batch
                if batch_idx == 0:
                    image_location = image_grid(images, np.asarray(labels.cpu()), np.array(pred_labels.cpu()), args['output_path'])
                    print(f'[INFO] image location: {image_location}')

        if args['show_metrics']:
            acc = metric_acc.compute()
            print(f"\nAccuracy:        {acc:.3f}")
            confmat = metric_confmat.compute()
            print(f'True Positives:  {confmat[1, 1]}')
            print(f'True Negatives:  {confmat[0, 0]}')
            print(f'False Positives: {confmat[0, 1]}')
            print(f'False Negatives: {confmat[1, 0]}')
            precision = metric_precision.compute()
            print(f'Precision:       {precision[1]:.3f}')
            recall = metric_recall.compute()
            print(f'Recall:          {recall[1]:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference of Test Images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=pathlib.Path, required=True, help='path to trained .pth model')
    parser.add_argument('--dataset-path', type=pathlib.Path, default=os.path.join(config.DATASET_PATH), metavar='PATH', help='path to test dataset')
    parser.add_argument('--batch', type=int, default=config.PRED_BATCH_SIZE, help='batch size')
    parser.add_argument('--show-metrics', default=True, action='store_true', help='print inference metrics when testing a dataset batch')
    parser.add_argument('--image-path', type=pathlib.Path, metavar='PATH', help='path to test images instead of dataset batch')
    parser.add_argument('--image-url', type=str, metavar='URL', help='URL to test images instead of dataset batch')
    parser.add_argument('--output-path', type=pathlib.Path, default='output', metavar='PATH', help='output path')
    args = vars(parser.parse_args())

    inference(args)
