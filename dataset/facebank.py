import torchvision.transforms as transforms
import torch.utils.data as data
import numpy as np
import cv2
import os
import torch


def img_loader(path):
    try:
        img = cv2.imread(path)
        if len(img.shape) == 2:
            img = np.stack([img] * 3, 2)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except IOError:
        print('Cannot load image ' + path)


class FaceBank(data.Dataset):
    def __init__(self, root, image_label_list, transform=None, loader=img_loader):

        self.root = root
        self.transform = transform
        self.loader = loader

        image_list = []
        label_list = []
        for info in image_label_list:
            image_path, label_name = info.split('\t')
            image_list.append(image_path)
            label_list.append(int(label_name))

        self.image_list = image_list
        self.label_list = label_list
        self.class_nums = len(np.unique(self.label_list))

    def __getitem__(self, index):
        img_path = self.image_list[index]
        label = self.label_list[index]

        img = self.loader(os.path.join(self.root, img_path))

        # random flip with ratio of 0.5
        flip = np.random.choice(2) * 2 - 1
        if flip == 1:
            img = cv2.flip(img, 1)

        if self.transform is not None:
            img = self.transform(img)
        else:
            img = torch.from_numpy(img)

        return img, label

    def __len__(self):
        return len(self.image_list)


if __name__ == '__main__':
    root = r'E:\Dataset\Human_Face_Dataset\facebank-112x112'
    file_list = r'E:\Dataset\Human_Face_Dataset\facebank-112x112.list'

    transform = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255] -> [0.0,1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(
            0.5, 0.5, 0.5))  # range [0.0, 1.0] -> [-1.0,1.0]
    ])
    dataset = FaceBank(root, file_list, transform=transform)
    trainloader = data.DataLoader(
        dataset, batch_size=64, shuffle=True, num_workers=2, drop_last=False)
    print(len(dataset))
    for data in trainloader:
        print(data[0].shape)
