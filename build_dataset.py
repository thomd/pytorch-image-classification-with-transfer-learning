import config
from imutils import paths
import numpy as np
import shutil
import argparse
import pathlib
import os

def copy_images(image_paths, folder):
    dataset_folder = os.path.join(args['dataset_path'], folder)
    print(f'[INFO] moving {len(image_paths)} images into {dataset_folder}')
    if not os.path.exists(dataset_folder):
        os.makedirs(dataset_folder)

    for path in image_paths:
        image_name = path.split(os.path.sep)[-1]
        label = path.split(os.path.sep)[-2]
        label_folder = os.path.join(dataset_folder, label)
        if not os.path.exists(label_folder):
            os.makedirs(label_folder)
        destination = os.path.join(label_folder, image_name)
        shutil.copy(path, destination)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create Dataset.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--images-path', type=pathlib.Path, default=config.IMAGES_PATH, help='path to image data')
    parser.add_argument('--dataset-path', type=pathlib.Path, default=config.DATASET_PATH, help='path to dataset')
    parser.add_argument('--shuffle', default=True, type=bool, help='shuffle images')
    parser.add_argument('--test-split', default=config.TEST_SPLIT, type=float, help='test split')
    parser.add_argument('--validation-split', default=config.VAL_SPLIT, type=float, help='validation split')
    args = vars(parser.parse_args())

    print('[INFO] loading images ...')
    image_paths = list(paths.list_images(args['images_path']))
    if args['shuffle']:
        np.random.shuffle(image_paths)

    test_paths_len = int(len(image_paths) * args['test_split'])
    val_paths_len = int(len(image_paths) * args['validation_split'])
    train_paths_len = len(image_paths) - test_paths_len - val_paths_len

    test_paths = image_paths[train_paths_len+val_paths_len:]
    val_paths = image_paths[train_paths_len:train_paths_len+val_paths_len]
    train_paths = image_paths[:train_paths_len]

    copy_images(train_paths, config.TRAIN)
    copy_images(val_paths, config.VAL)
    copy_images(test_paths, config.TEST)




