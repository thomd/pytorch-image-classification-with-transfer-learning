import config
from pathlib import Path
import numpy as np
import shutil
import argparse
import pathlib
import os

def copy_images(image_paths, folder):
    dataset_folder = os.path.join(args['dataset_path'], folder)
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


def tree(dir_path, prefix=''):
    space, branch, tee, last =  '    ', '│   ', '├── ', '└── '
    contents = [p for p in dir_path.iterdir() if p.is_dir()]
    pointers = [tee] * (len(contents) - 1) + [last]
    for pointer, path in zip(pointers, contents):
        count = len([p for p in path.iterdir() if p.is_file()])
        yield prefix + pointer + path.name + (f' ({count} images)' if count > 0 else '')
        if path.is_dir():
            extension = branch if pointer == tee else space
            yield from tree(path, prefix=prefix+extension)


def list_images(image_path):
    image_types = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    image_paths = [(Path(p.parent).name, str(p)) for p in Path('temp').glob('*/*.*') if p.suffix in image_types]
    label_paths = {}
    for i in image_paths:
        label_paths.setdefault(i[0],[]).append(i[1])
    return label_paths


def build_dataset(args):
    if args['show_tree']:
        dataset_path = Path(args['dataset_path'])
        if dataset_path.exists():
            print(dataset_path.name)
            for line in tree(dataset_path):
                print(line)
        else:
            print(f"dataset folder '{str(dataset_path)}' does not exist.")
        return

    label_paths = list_images(args['images_path'])

    for label in label_paths.keys():
        print(f'[INFO] label: {label}')
        image_paths = label_paths.get(label)

        if args['shuffle']:
            print('[INFO] shuffling image paths ...')
            np.random.shuffle(image_paths)

        test_paths_len = int(len(image_paths) * args['test_split'])
        val_paths_len = int(len(image_paths) * args['val_split'])
        train_paths_len = len(image_paths) - test_paths_len - val_paths_len

        test_paths = image_paths[-test_paths_len:]
        val_paths = image_paths[train_paths_len:-test_paths_len]
        train_paths = image_paths[:train_paths_len]

        copy_images(train_paths, config.TRAIN)
        copy_images(val_paths, config.VAL)
        copy_images(test_paths, config.TEST)

    dataset_path = Path(args['dataset_path'])
    print(f'\n{dataset_path.name}')
    for line in tree(dataset_path):
        print(line)


if __name__ == '__main__':
    class CustomFormatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter):
        pass

    parser = argparse.ArgumentParser(description='''Create Dataset.

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
''', formatter_class=CustomFormatter)
    parser.add_argument('--images-path', type=pathlib.Path, default=config.IMAGES_PATH, metavar='PATH', help='path to image data')
    parser.add_argument('--dataset-path', type=pathlib.Path, default=config.DATASET_PATH, metavar='PATH', help='path to dataset')
    parser.add_argument('--shuffle', default=True, action='store_true', help='shuffle images')
    parser.add_argument('--test-split', default=config.TEST_SPLIT, type=float, metavar='VAL', help='test split')
    parser.add_argument('--val-split', default=config.VAL_SPLIT, type=float, metavar='VAL', help='validation split')
    parser.add_argument('--show-tree', action='store_true', help='show dataset tree with number of images and exit')
    args = vars(parser.parse_args())

    build_dataset(args)



