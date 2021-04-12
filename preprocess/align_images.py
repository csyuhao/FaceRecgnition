import os
import sys
import torch
import argparse
import numpy as np
from PIL import Image
from mtcnn.mtcnn import MTCNN
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mtcnn.utils.align_trans import get_reference_facial_points, warp_and_crop_face


def collate_pil(x):
    out_x, out_y = [], []
    for xx, yy in x:
        out_x.append(xx)
        out_y.append(yy)
    return out_x, out_y


def save_landmark(landmark, save_path):
    """Save Landmark to path

    Arguments:
        landmark {float} -- landmarks
        save_path {str} -- Save path for extracted face image. (default: {None})

    Returns:
        None
    """
    with open(save_path, 'w+') as f:
        for (x, y) in landmark:
            f.write('{}\t{}\n'.format(x, y))

    return landmark


def main(args):

    workers = 0 if os.name == 'nt' else 8
    cropped_dataset = '{}-{}x{}'.format(args.output_dir, args.img_size, args.img_size)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    mtcnn = MTCNN(
        keep_all=False,
        select_largest=False,
        selection_method='center_weighted_size',
        device=device
    )

    dataset = datasets.ImageFolder(
        args.input_dir, transform=transforms.Resize((512, 512)))
    dataset.samples = [
        (p, p.replace(args.input_dir, cropped_dataset))
        for p, _ in dataset.samples
    ]
    loader = DataLoader(
        dataset,
        num_workers=workers,
        batch_size=args.batch_size,
        collate_fn=collate_pil
    )

    scale = args.img_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    for i, (images, paths) in enumerate(loader):
        batch_boxes, _, batch_landmarks = mtcnn.detect_only_face(images, landmarks=True)

        for idx, (boxes, landmarks, img_path) in enumerate(zip(batch_boxes, batch_landmarks, paths)):
            if boxes is None:
                continue
            facial5points = landmarks[0]
            warped_face, _landmarks = warp_and_crop_face(np.array(images[idx]), facial5points, reference, crop_size=(args.img_size, args.img_size))
            img_warped = Image.fromarray(warped_face)
            if img_path.split('.')[-1].lower() not in ['jpg', 'jpeg']:    # not from jpg
                img_path = '.'.join(img_path.split('.')[:-1]) + '.jpg'
            os.makedirs(os.path.dirname(img_path) + "/", exist_ok=True)
            img_warped.save(img_path)

            if args.save_landmarks:
                save_landmark(_landmarks, '.'.join(img_path.split('.')[:-1]) + '.txt')


def parse_args(argv):
    parser = argparse.ArgumentParser('Cropped Dataset')
    parser.add_argument('--input_dir', type=str, default=r'images', help='The path to the uncropped dataset')
    parser.add_argument('--output_dir', type=str, default=r'output', help='The path to the cropped dataset')
    parser.add_argument('--save_landmarks', action='store_true', help='Whether to store landmarks')
    parser.add_argument('--img_size', type=int, default=112, help='The cropped image size')
    parser.add_argument('--batch_size', type=int, default=64, help='The batch size')
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
