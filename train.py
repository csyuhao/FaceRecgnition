import os
import sys
import time
import torch
import argparse
import numpy as np
import torch.optim as optim
from dataset.lfw import LFW
from datetime import datetime
from torch.nn import DataParallel
from util.logging import init_log
from torchvision import transforms
from torch.optim import lr_scheduler
from util.visualize import Visualizer
from torch.utils.data import DataLoader
from dataset.webface import CASIAWebFace
from loss.FaceNetLoss import InnerProduct
from backbone.mobilefacenet import MobileFaceNet
from loss.ArcFaceLossMargin import ArcFaceLossMargin
from loss.CosFaceLossMargin import CosineMarginProduct
from loss.SphereFaceMarigin import SphereMarginProduct
from eval_lfw import evaluation_10_fold, getFeatureFromTorch
from backbone.resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.irse import IR_50, IR_SE_50, IR_101, IR_SE_101, IR_152, IR_SE_152
from backbone.vgg import VGG11, VGG11_BN, VGG13, VGG13_BN, VGG16, VGG16_BN, VGG19, VGG19_BN


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
    trans = transforms.Compose([
        transforms.ToTensor(),  # range [0, 255.0] => [0.0, 1.0]
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(
            0.5, 0.5, 0.5))  # range [0.0, 1.0] => [-1.0, 1.0]
    ])
    # training dataset
    train_dataset = CASIAWebFace(
        args.train_root, args.train_file_list, transform=trans)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=8, drop_last=False)
    # testing dataset
    test_dataset = LFW(args.lfw_test_root, args.lfw_file_list, transform=trans)
    test_loader = DataLoader(test_dataset, batch_size=128,
                             shuffle=False, num_workers=4, drop_last=False)

    # define backbone
    if args.backbone.upper() == 'RESNET_50':
        net = ResNet_50([112, 112])
    elif args.backbone.upper() == 'RESNET101':
        net = ResNet_101([112, 112])
    elif args.backbone.upper() == 'RESNET152':
        net = ResNet_152([112, 112])
    elif args.backbone.upper() == 'IR_50':
        net = IR_50([112, 112])
    elif args.backbone.upper() == 'IR_SE_50':
        net = IR_SE_50([112, 112])
    elif args.backbone.upper() == 'IR_101':
        net = IR_101([112, 112])
    elif args.backbone.upper() == 'IR_SE_101':
        net = IR_SE_101([112, 112])
    elif args.backbone.upper() == 'IR_152':
        net = IR_152([112, 112])
    elif args.backbone.upper() == 'IR_SE_152':
        net = IR_SE_152([112, 112])
    elif args.backbone.upper() == 'VGG11':
        net = VGG11()
    elif args.backbone.upper() == 'VGG11_BN':
        net = VGG11_BN()
    elif args.backbone.upper() == 'VGG13':
        net = VGG13()
    elif args.backbone.upper() == 'VGG13_BN':
        net = VGG13_BN()
    elif args.backbone.upper() == 'VGG16':
        net = VGG16()
    elif args.backbone.upper() == 'VGG16_BN':
        net = VGG16_BN()
    elif args.backbone.upper() == 'VGG19':
        net = VGG19()
    elif args.backbone.upper() == 'VGG19_BN':
        net = VGG19_BN()
    elif args.backbone.upper() == 'MOBILEFACENET':
        net = MobileFaceNet(feature_dim=512)
    else:
        raise NameError(args.backbone, ' is not availabe!')

    # define margin
    if args.margin_type.upper() == 'ARCFACE':
        margin = ArcFaceLossMargin(in_feature=args.feature_dim, out_feature=train_dataset.class_nums, s=args.scale_size)
    elif args.margin_type.upper() == 'COSFACE':
        margin = CosineMarginProduct(in_feature=args.feature_dim, out_feature=train_dataset.class_nums, s=args.scale_size)
    elif args.margin_type.upper() == 'FACENET':
        margin = InnerProduct(in_feature=args.feature_dim,
                              out_feature=train_dataset.class_nums)
    elif args.margin_type.upper() == 'SPHEREFACE':
        margin = SphereMarginProduct(
            in_feature=args.feature_dim, out_feature=train_dataset.class_nums)
    else:
        raise NameError(args.margin_type, 'is not avaliable!')

    if args.resume:
        print('Resuming the model parameters from: ',
              args.net_path, args.margin_path)
        net.load_state_dict(torch.load(args.net_path)['net_state_dict'])
        margin.load_state_dict(torch.load(args.margin_path)[
                               'net_state_dict'])

    # define optimizers for different layer
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer_ft = optim.SGD([
        {'params': net.parameters(), 'weight_decay': 5e-4},
        {'params': margin.parameters(), 'weight_decay': 5e-4},
    ], lr=0.1, momentum=0.9, nesterov=True)
    milestone_ratios = [0.33, 0.60, 0.88]
    exp_lr_scheduler = lr_scheduler.MultiStepLR(
        optimizer_ft, milestones=[int(r * args.total_epoch) for r in milestone_ratios], gamma=0.1)

    if multi_gpus:
        net = DataParallel(net).to(device)
        margin = DataParallel(margin).to(device)
    else:
        net = net.to(device)
        margin = margin.to(device)

    best_lfw_acc, best_lfw_iters, total_iters = 0., 0, 0
    vis = Visualizer(env=args.model_pre + args.backbone + '_' + args.margin_type.upper())
    for epoch in range(1, args.total_epoch + 1):

        # train model
        _print('Train Epoch: {}/{} ...'.format(epoch, args.total_epoch))
        net.train()

        since = time.time()
        for data in train_loader:
            img, label = data[0].to(device), data[1].to(device)
            optimizer_ft.zero_grad()

            raw_logits = net(img)
            output = margin(raw_logits, label)
            total_loss = criterion(output, label)
            total_loss.backward()
            optimizer_ft.step()

            total_iters += 1
            if total_iters % 100 == 0:
                # current training accuracy
                _, predict = torch.max(output.data, 1)
                total = label.size(0)
                correct = (np.array(predict.cpu()) ==
                           np.array(label.data.cpu())).sum()
                time_cur = (time.time() - since) / 100
                since = time.time()
                vis.plot_curves({'softmax loss': total_loss.item()}, iters=total_iters, title='train loss',
                                xlabel='iters', ylabel='train loss')
                vis.plot_curves({'train accuracy': correct / total}, iters=total_iters, title='train accuracy', xlabel='iters',
                                ylabel='train accuracy')

                _print("Iters: {:0>6d}/[{:0>2d}], loss: {:.4f}, train_accuracy: {:.4f}, time: {:.2f} s/iter, learning rate: {}".format(
                    total_iters, epoch, total_loss.item(), correct/total, time_cur, optimizer_ft.param_groups[0]['lr']))

            # save model
            if total_iters % args.save_freq == 0:
                msg = 'Saving checkpoint: {}'.format(total_iters)
                _print(msg)
                if multi_gpus:
                    net_state_dict = net.module.state_dict()
                    margin_state_dict = margin.module.state_dict()
                else:
                    net_state_dict = net.state_dict()
                    margin_state_dict = margin.state_dict()
                if not os.path.exists(save_dir):
                    os.mkdir(save_dir)
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': net_state_dict
                }, os.path.join(save_dir, 'Iter_%06d_net.ckpt' % total_iters))
                torch.save({
                    'iters': total_iters,
                    'net_state_dict': margin_state_dict
                }, os.path.join(save_dir, 'Iter_%06d_margin.ckpt' % total_iters))

            # test accuracy
            if total_iters % args.test_freq == 0:

                # test model on lfw
                net.eval()
                getFeatureFromTorch('result/cur_lfw_result.mat', net, device, test_dataset, test_loader)
                lfw_accs = evaluation_10_fold('result/cur_lfw_result.mat')
                _print('LFW Ave Accuracy: {:.4f}'.format(np.mean(lfw_accs) * 100))
                if best_lfw_acc <= np.mean(lfw_accs) * 100:
                    best_lfw_acc = np.mean(lfw_accs) * 100
                    best_lfw_iters = total_iters

                _print('Current Best Accuracy: LFW: {:.4f} in iters: {}'.format(
                    best_lfw_acc, best_lfw_iters))

                vis.plot_curves({'lfw': np.mean(lfw_accs)}, iters=total_iters, title='test accuracy', xlabel='iters', ylabel='test accuracy')
                net.train()
        exp_lr_scheduler.step()
    _print('Finally Best Accuracy: LFW: {:.4f} in iters: {},'.format(best_lfw_acc, best_lfw_iters))
    vis.save()
    print('finishing training')


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description='PyTorch for deep face recognition')
    parser.add_argument('--train_root', type=str,
                        default=r'datasets\CASIA-WebFace-112x112', help='train image root')
    parser.add_argument('--train_file_list', type=str,
                        default=r'datasets\CASIA-WebFace-112x112.list', help='train list')
    parser.add_argument('--lfw_test_root', type=str,
                        default=r'datasets\lfw-112x112', help='lfw image root')
    parser.add_argument('--lfw_file_list', type=str,
                        default=r'datasets\pairs.txt', help='lfw pair file list')

    parser.add_argument('--backbone', type=str, default='IR_SE_50',
                        help='IR_SE_50, IR_SE_101, IR_SE_152...')
    parser.add_argument('--margin_type', type=str, default='ArcFace',
                        help='ArcFace, CosFace, SphereFace, VGGFace, FaceNet')
    parser.add_argument('--feature_dim', type=int, default=512,
                        help='feature dimension, 128 or 512')
    parser.add_argument('--scale_size', type=float,
                        default=32.0, help='scale size')
    parser.add_argument('--batch_size', type=int,
                        default=200, help='batch size')
    parser.add_argument('--total_epoch', type=int,
                        default=18, help='total epochs')

    parser.add_argument('--save_freq', type=int,
                        default=3000, help='save frequency')
    parser.add_argument('--test_freq', type=int,
                        default=3000, help='test frequency')
    parser.add_argument('--resume', type=int,
                        default=False, help='resume model')
    parser.add_argument('--net_path', type=str,
                        default='', help='resume model')
    parser.add_argument('--margin_path', type=str,
                        default='', help='resume model')
    parser.add_argument('--save_dir', type=str,
                        default='model', help='model save dir')
    parser.add_argument('--model_pre', type=str,
                        default='FRS_', help='model prefix')
    parser.add_argument('--gpus', type=str,
                        default='0', help='model prefix')
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))
