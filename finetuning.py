import os
import sys
import time
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch.nn import DataParallel
from util.logging import init_log
from torchvision import transforms
from torch.optim import lr_scheduler
from dataset.facebank import FaceBank
from util.visualize import Visualizer
from torch.utils.data import DataLoader
from backbone.mobilefacenet import MobileFaceNet
from backbone.resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.irse import IR_50, IR_SE_50, IR_101, IR_SE_101, IR_152, IR_SE_152
from backbone.vgg import VGG11, VGG11_BN, VGG13, VGG13_BN, VGG16, VGG16_BN, VGG19, VGG19_BN


class FaceRecognizer(nn.Module):

    def __init__(self, feature, classnum):
        super(FaceRecognizer, self).__init__()
        self.feature = feature
        self.classifier = nn.Sequential(*[
            nn.Linear(512, 256),
            nn.Tanh(),
            nn.Linear(256, classnum)
        ])

    def load_feature(self, file_path):
        self.feature.load_state_dict(torch.load(file_path)['net_state_dict'])

    def train(self, mode=True):
        self.feature.eval()
        self.training = mode
        for module in self.classifier.children():
            module.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def forward(self, x):
        out = self.feature(x)
        logit = self.classifier(out)
        return logit


def main(args):
    seed = 117
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # gpu initialize
    multi_gpus = False
    if len(args.gpus.split(',')) > 1:
        multi_gpus = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # log initialize
    save_dir = os.path.join(args.save_dir, args.model_pre +
                            args.backbone.upper() + '_' +
                            args.margin_type.upper() + '_' +
                            datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_dir):
        raise NameError('Model directionary exists!')
    os.makedirs(save_dir)
    logging = init_log(save_dir)
    _print = logging.info

    # dataset loader
    # prepare dataset
    with open(args.train_file_list) as f:
        img_label_list = f.read().splitlines()
    np.random.shuffle(img_label_list)
    train_len = len(img_label_list) - len(img_label_list) // 10
    trans = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255.0] => [0.0, 1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(
            0.5, 0.5, 0.5))  # range [0.0, 1.0] => [-1.0, 1.0]
    ])
    # training dataset
    train_dataset = FaceBank(
        args.train_root, img_label_list[:train_len], transform=trans)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, drop_last=False)
    # testing dataset
    test_dataset = FaceBank(args.train_root, img_label_list[train_len:], transform=trans)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=4, drop_last=False)

    # define backbone
    if args.backbone.upper() == 'RESNET_50':
        feature = ResNet_50([112, 112])
    elif args.backbone.upper() == 'RESNET101':
        feature = ResNet_101([112, 112])
    elif args.backbone.upper() == 'RESNET152':
        feature = ResNet_152([112, 112])
    elif args.backbone.upper() == 'IR_50':
        feature = IR_50([112, 112])
    elif args.backbone.upper() == 'IR_SE_50':
        feature = IR_SE_50([112, 112])
    elif args.backbone.upper() == 'IR_101':
        feature = IR_101([112, 112])
    elif args.backbone.upper() == 'IR_SE_101':
        feature = IR_SE_101([112, 112])
    elif args.backbone.upper() == 'IR_152':
        feature = IR_152([112, 112])
    elif args.backbone.upper() == 'IR_SE_152':
        feature = IR_SE_152([112, 112])
    elif args.backbone.upper() == 'VGG11':
        feature = VGG11()
    elif args.backbone.upper() == 'VGG11_BN':
        feature = VGG11_BN()
    elif args.backbone.upper() == 'VGG13':
        feature = VGG13()
    elif args.backbone.upper() == 'VGG13_BN':
        feature = VGG13_BN()
    elif args.backbone.upper() == 'VGG16':
        feature = VGG16()
    elif args.backbone.upper() == 'VGG16_BN':
        feature = VGG16_BN()
    elif args.backbone.upper() == 'VGG19':
        feature = VGG19()
    elif args.backbone.upper() == 'VGG19_BN':
        feature = VGG19_BN()
    elif args.backbone.upper() == 'MOBILEFACENET':
        feature = MobileFaceNet(feature_dim=512)
    else:
        raise NameError(args.backbone, ' is not availabe!')

    # define classifer
    recognizer = FaceRecognizer(feature, train_dataset.class_nums)
    recognizer.load_feature(args.backbone_path)

    if args.resume:
        print('Resuming the model parameters from: ', args.net_path)
        recognizer.load_state_dict(torch.load(args.net_path)['recognizer_state_dict'])

    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD(recognizer.classifier.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    milestone_ratios = [0.33, 0.60, 0.88]
    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer_ft, milestones=[int(r * args.total_epoch) for r in milestone_ratios], gamma=0.1)

    if multi_gpus:
        recognizer = DataParallel(recognizer).to(device)
    else:
        recognizer = recognizer.to(device)

    best_acc, best_loss, best_iters, total_iters = 0., 0., 0, 0
    vis = Visualizer(env=args.model_pre + args.backbone + '_' + args.margin_type.upper())
    for epoch in range(1, args.total_epoch + 1):

        # train model
        _print('Train Epoch: {}/{} ...'.format(epoch, args.total_epoch))
        recognizer.train()

        since = time.time()
        for data in train_loader:
            img, label = data[0].to(device), data[1].to(device)
            optimizer_ft.zero_grad()

            logits = recognizer(img)
            loss = criterion(logits, label)
            loss.backward()
            optimizer_ft.step()

            total_iters += 1
            if total_iters % 10 == 0:
                # current training accuracy
                _, predict = torch.max(logits.data, 1)
                total = label.size(0)
                correct = (np.array(predict.cpu()) ==
                           np.array(label.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                vis.plot_curves({'softmax loss': loss.item()}, iters=total_iters, title='train loss',
                                xlabel='iters', ylabel='train loss')
                vis.plot_curves({'train accuracy': correct / total}, iters=total_iters, title='train accuracy', xlabel='iters',
                                ylabel='train accuracy')

                _print("Iters: {:0>6d}/[{:0>2d}], loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(
                    total_iters, epoch, loss.item(), correct/total, time_cur, optimizer_ft.param_groups[0]['lr']))

            # save model
            if total_iters % args.save_freq == 0:
                msg = 'Saving checkpoint: {}'.format(total_iters)
                _print(msg)
                if multi_gpus:
                    recognizer_state_dict = recognizer.module.state_dict()
                else:
                    recognizer_state_dict = recognizer.state_dict()
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save({
                    'iters': total_iters,
                    'recognizer_state_dict': recognizer_state_dict
                }, os.path.join(save_dir, 'Iter_%06d_recognizer.ckpt' % total_iters))

            # test accuracy
            if total_iters % args.test_freq == 0:

                # test model on testing dataset
                recognizer.eval()
                test_acc, test_loss = 0., 0.
                with torch.no_grad():
                    for data in test_loader:
                        img, label = data[0].to(device), data[1].to(device)
                        logits = recognizer(img)
                        loss = criterion(logits, label)
                        test_loss += loss.detach().cpu().numpy() * img.shape[0]
                        _, predict = torch.max(logits.data, 1)
                        test_acc += (np.array(predict.cpu()) == np.array(label.data.cpu())).sum()
                    test_acc /= len(test_dataset)
                    test_loss /= len(test_dataset)
                _print('Testing Ave Accuracy: {:.4f}, Test Ave Loss: {:.4f}'.format(test_acc * 100, test_loss))
                if best_acc <= test_acc * 100:
                    best_acc = test_acc * 100
                    best_loss = test_loss
                    best_iters = total_iters

                _print('Current Best Accuracy: {:.4f}, Current Best Loss: {:.4f} in iters: {}'.format(
                    best_acc, best_loss, best_iters))

                vis.plot_curves({'testing': best_acc}, iters=total_iters, title='test accuracy', xlabel='iters', ylabel='test accuracy')
                recognizer.train()
        exp_lr_scheduler.step()
    _print('Finally Best Accuracy: {:.4f}, Best Loss: {:.4f} in iters: {},'.format(best_acc, best_loss, best_iters))
    vis.save()
    print('finishing training')


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='PyTorch for deep face recognition')
    parser.add_argument('--train_root', type=str,
                        default=r'Dataset\Human_Face_Dataset\facebank-112x112', help='train image root')
    parser.add_argument('--train_file_list', type=str,
                        default=r'Dataset\Human_Face_Dataset\facebank-112x112.list', help='train list')

    parser.add_argument('--backbone', type=str, default='IR_SE_50',
                        help='IR_SE_50, IR_SE_101, IR_SE_152...')
    parser.add_argument('--margin_type', type=str, default='ArcFace',
                        help='The margin type')
    parser.add_argument('--batch_size', type=int,
                        default=200, help='batch size')
    parser.add_argument('--total_epoch', type=int,
                        default=18, help='total epochs')

    parser.add_argument('--save_freq', type=int,
                        default=300, help='save frequency')
    parser.add_argument('--test_freq', type=int,
                        default=300, help='test frequency')
    parser.add_argument('--backbone_path', type=str,
                        default='', help='backbone model')
    parser.add_argument('--resume', type=int,
                        default=False, help='resume model')
    parser.add_argument('--net_path', type=str,
                        default='', help='resume model')
    parser.add_argument('--save_dir', type=str,
                        default='finetune_model', help='model save dir')
    parser.add_argument('--model_pre', type=str,
                        default='FRS_', help='model prefix')
    parser.add_argument('--gpus', type=str,
                        default='0', help='model prefix')
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
