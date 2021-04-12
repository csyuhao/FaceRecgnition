import os
import sys
import argparse


def dataset_list(dataset_path, dataset_list):
    label_list = os.listdir(dataset_path)
    f = open(dataset_list, 'w')
    k = 0
    for i in label_list:
        label_path = os.path.join(dataset_path, i)
        if os.listdir(label_path):
            image_list = os.listdir(label_path)
            for j in image_list:
                image_path = os.path.join(label_path, j)
                f.write(image_path.replace(dataset_path + '\\', '') + '\t' + str(k) + '\n')
        k = k + 1
    f.close()


def parse_args(argv):
    parser = argparse.ArgumentParser('Train Dataset List')
    parser.add_argument('--dataset_path', type=str, required=True, help='The path to the cropped dataset')
    parser.add_argument('--dataset_list', type=str, required=True, help='The path to the dataset list')
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    dataset_list(args.dataset_path, args.dataset_list)
