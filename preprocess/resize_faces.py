import os
import cv2
import sys
import argparse


def resize(dataset_path, dataset_resized):

    if dataset_resized is None:
        dataset_resized = dataset_path + '_clipped'

    for dir_name in os.listdir(dataset_path):
        for filename in os.listdir(os.path.join(dataset_path, dir_name)):
            filepath = os.path.join(os.path.join(dataset_path, dir_name), filename)

            image = cv2.imread(filepath)
            height, width = image.shape[:2]
            if width != height:
                center = width // 2
                left = center - height // 2
                right = center + height // 2
                image = image[:, left:right]
            n_filepath = filepath.replace(dataset_path, dataset_resized)
            os.makedirs(os.path.dirname(n_filepath) + "/", exist_ok=True)
            cv2.imwrite(n_filepath, image)


def parse_args(argv):
    parser = argparse.ArgumentParser('Train Dataset List')
    parser.add_argument('--dataset_path', type=str, required=True, help='The path to the cropped dataset')
    parser.add_argument('--dataset_resized', type=str, help='The path to the resized dataset')
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    resize(args.dataset_path, args.dataset_resized)
