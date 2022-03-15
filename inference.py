import config
from utils import create_image_grid
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
        transforms.ToTensor()
    ])

    print('[INFO] loading the dataset...')
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

    # grab a batch of test data
    batch = next(iter(test_loader))
    (images, labels) = (batch[0], batch[1])

    # switch off autograd
    with torch.no_grad():

        # send the images to the device
        images = images.to(config.DEVICE)

        # make the predictions
        print('[INFO] performing inference...')
        preds = model(images)

        true_labels = np.asarray(labels)
        pred_labels = np.array([pred.argmax() for pred in preds.cpu()])

        if args['show_metrics']:
            print(f'Truth:     {true_labels}')
            print(f'Predicion: {pred_labels}')

        create_image_grid(images, true_labels, pred_labels, args['inference_path'], nrow=8)
        print(f'[INFO] image location: {args["inference_path"]}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference of Test Images', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', type=pathlib.Path, required=True, help='path to trained model model')
    parser.add_argument('--dataset-path', type=pathlib.Path, default=os.path.join(config.DATASET_PATH, config.TEST), help='path to dataset with test images')
    parser.add_argument('--inference-path', type=pathlib.Path, default='output/inference.png', help='path to inferences image')
    parser.add_argument('--show-metrics', type=bool, default=True, help='print inference metrics')
    parser.add_argument('--batch', type=int, default=config.PRED_BATCH_SIZE, help='batch size')
    args = vars(parser.parse_args())

    inference(args)
