import os
import sys
import cv2
import time
import torch
import argparse
import numpy as np
from datetime import datetime
from mtcnn.mtcnn import MTCNN
import torch.nn.functional as F
from model import FaceRecognizer
from PIL import Image, ImageDraw, ImageFont
from backbone.mobilefacenet import MobileFaceNet
from backbone.resnet import ResNet_50, ResNet_101, ResNet_152
from backbone.irse import IR_50, IR_SE_50, IR_101, IR_SE_101, IR_152, IR_SE_152
from mtcnn.utils.align_trans import get_reference_facial_points, warp_and_crop_face
from backbone.vgg import VGG11, VGG11_BN, VGG13, VGG13_BN, VGG16, VGG16_BN, VGG19, VGG19_BN


def load_model(model_dir, model_name, classnum=153, prefix='FRS', best_epoch=None, device='cpu'):
    backbone_name, margin_name = model_name.split('-')
    backbone = None
    if backbone_name.upper() == 'RESNET_50':
        backbone = ResNet_50([112, 112])
    elif backbone_name.upper() == 'RESNET101':
        backbone = ResNet_101([112, 112])
    elif backbone_name.upper() == 'RESNET152':
        backbone = ResNet_152([112, 112])
    elif backbone_name.upper() == 'IR_50':
        backbone = IR_50([112, 112])
    elif backbone_name.upper() == 'IR_SE_50':
        backbone = IR_SE_50([112, 112])
    elif backbone_name.upper() == 'IR_101':
        backbone = IR_101([112, 112])
    elif backbone_name.upper() == 'IR_SE_101':
        backbone = IR_SE_101([112, 112])
    elif backbone_name.upper() == 'IR_152':
        backbone = IR_152([112, 112])
    elif backbone_name.upper() == 'IR_SE_152':
        backbone = IR_SE_152([112, 112])
    elif backbone_name.upper() == 'VGG11':
        backbone = VGG11()
    elif backbone_name.upper() == 'VGG11_BN':
        backbone = VGG11_BN()
    elif backbone_name.upper() == 'VGG13':
        backbone = VGG13()
    elif backbone_name.upper() == 'VGG13_BN':
        backbone = VGG13_BN()
    elif backbone_name.upper() == 'VGG16':
        backbone = VGG16()
    elif backbone_name.upper() == 'VGG16_BN':
        backbone = VGG16_BN()
    elif backbone_name.upper() == 'VGG19':
        backbone = VGG19()
    elif backbone_name.upper() == 'VGG19_BN':
        backbone = VGG19_BN()
    elif backbone_name.upper() == 'MOBILEFACENET':
        backbone = MobileFaceNet(feature_dim=512)
    else:
        raise NameError(backbone_name, ' is not availabe!')

    assert backbone is not None, 'The backbone cannot find'

    recognizer = FaceRecognizer(backbone, classnum).to(device).eval()

    for param in recognizer.parameters():
        param.grad = None

    model_name = '%s_%s_%s' % (prefix, backbone_name, margin_name)
    for name in os.listdir(model_dir):
        if name.startswith(model_name) is True:
            model_name = name
            break

    model_path = os.path.join(model_dir, model_name)
    last_state = None
    for model_name in os.listdir(model_path):
        if model_name.endswith('.ckpt'):
            state = torch.load(os.path.join(model_path, model_name))
            if best_epoch is not None and state['iters'] == best_epoch:
                recognizer.load_state_dict(state['recognizer_state_dict'])
                return recognizer
            last_state = state
    recognizer.load_state_dict(last_state['recognizer_state_dict'])
    return recognizer


def load_name_list(name_list):
    idx_name = {}
    with open(name_list, 'r') as f:
        for line in f.readlines():
            path, idx = line.strip().split('\t')
            name = os.path.split(path)[0]
            idx_name[int(idx)] = name
    return idx_name


def main(args):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    cam.set(3, 1024)
    cam.set(4, 1024)
    cv2.namedWindow('Face Recognition')
    success, frame = cam.read()
    saved_frame = frame.copy()

    save_path = r'video_testing\{}\{}'.format(args.model_name, datetime.now().strftime('%Y%m%d_%H%M%S'))
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 1
    size = (int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    annotated_out = cv2.VideoWriter(os.path.join(save_path, 'annotated.mp4'), fourcc, fps, size)
    unannotated_out = cv2.VideoWriter(os.path.join(save_path, 'unannotated.mp4'), fourcc, fps, size)
    font = ImageFont.truetype("consola.ttf", 18, encoding="unic")

    mtcnn = MTCNN(
        image_size=(160, 160),
        keep_all=False,
        select_largest=False,
        selection_method='center_weighted_size',
        device=device
    )

    name_list = load_name_list(args.name_list)
    recognizer = load_model(args.model_path, args.model_name, classnum=len(name_list), best_epoch=args.best_epoch, device=device)

    scale = args.img_size / 112.
    reference = get_reference_facial_points(default_square=True) * scale

    start_time = time.time()
    cnt = 0
    while success:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        boxes, _, landmarks = mtcnn.detect_only_face(frame, landmarks=True)

        if boxes is not None:
            facial5points = landmarks[0]
            warped_face, _ = warp_and_crop_face(np.array(frame), facial5points, reference, crop_size=(args.img_size, args.img_size))

            warped_face = warped_face.transpose(2, 0, 1)
            faces = torch.from_numpy(warped_face[np.newaxis, :]).to(device) / 255. * 2. - 1
            with torch.no_grad():
                logits = recognizer.forward(faces)
                probs = F.softmax(logits, dim=1)

            max_idx = torch.argmax(probs, dim=1).detach().cpu().item()
            prob = probs.topk(1, dim=1)[0].detach().cpu().item()
            name = name_list[max_idx]

            frame_draw = frame.copy()
            draw = ImageDraw.Draw(frame_draw)
            for box in boxes:
                draw.text((box.tolist()[0], box.tolist()[1] - 20), 'Id: %s Conf: %.4f' % (name, prob), (255, 0, 0), font=font)
                draw.rectangle(box.tolist(), outline=(255, 0, 0), width=6)

            frame = cv2.cvtColor(np.asarray(frame_draw), cv2.COLOR_RGB2BGR)

        annotated_out.write(frame)
        unannotated_out.write(saved_frame)

        cv2.imshow('Face Recognition', frame)
        end_time = time.time()
        if (end_time - start_time) > 20:
            break
        success, frame = cam.read()
        saved_frame = frame.copy()

    cam.release()
    annotated_out.release()
    unannotated_out.release()
    cv2.destroyAllWindows()


def parse_args(argv):
    parser = argparse.ArgumentParser(description='Video Testing')
    parser.add_argument('--model_path', type=str, required=True, help='The path to storing models')
    parser.add_argument('--backbone_name', type=str, required=True, help='The model name of recognizer (i.e., IR_SE_50-ArcFace)')
    parser.add_argument('--best_epoch', type=int, default=None, help='The epoch of trained models')
    parser.add_argument('--img_size', type=int, default=112, help='The size of cropped images')
    parser.add_argument('--name_list', type=str, default=r'E:\Dataset\Human_Face_Dataset\facebank-112x112.list')
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_args(sys.argv[1:]))
