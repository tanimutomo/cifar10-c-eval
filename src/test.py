import argparse
import numpy as np
import os
import pprint
import torch
import torchvision
import tqdm

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm

from utils import load_txt, accuracy, create_barplot, get_fname, AverageMeter
from models.resnet import ResNet56
from dataset import CIFAR10C

corruptions = load_txt('./src/corruptions.txt')


def main(opt):

    device = torch.device(opt.gpu_id)

    # model
    if opt.arch == 'resnet56':
        model = ResNet56()
    else:
        raise ValueError()
    model.load_state_dict(torch.load(opt.weight_path, map_location='cpu'))
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.49139968, 0.48215841, 0.44653091],
                             [0.24703223, 0.24348513, 0.26158784],)
    ])

    accs = dict()
    with tqdm(total=len(opt.corruptions), ncols=80) as pbar:
        for ci, cname in enumerate(opt.corruptions):
            # load dataset
            if cname == 'natural':
                dataset = datasets.CIFAR10(
                    os.path.join(opt.data_root, 'cifar10'),
                    train=False, transform=transform
                )
            else:
                dataset = CIFAR10C(
                    os.path.join(opt.data_root, 'cifar10-c'),
                    cname, transform=transform
                )
            loader = DataLoader(dataset, batch_size=opt.batch_size,
                                shuffle=False, num_workers=4)
            
            acc_meter = AverageMeter()
            with torch.no_grad():
                for itr, (x, y) in enumerate(loader):
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, dtype=torch.int64, non_blocking=True)

                    # calcurate clean loss and accuracy
                    z = model(x)
                    loss = F.cross_entropy(z, y)
                    acc, _ = accuracy(z, y, topk=(1, 5))
                    acc_meter.update(acc.item())

            accs[f'{cname}'] = acc_meter.avg

            pbar.set_postfix_str(f'{cname}: {acc_meter.avg:.2f}')
            pbar.update()
    
    pprint.pprint(accs)
    save_name = get_fname(opt.weight_path)
    create_barplot(accs, os.path.join('figs', save_name+'.png'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        '--arch',
        type=str, default='resnet56',
        help='model name'
    )
    parser.add_argument(
        '--weight_path',
        type=str, required=True,
        help='path to model weight',
    )
    parser.add_argument(
        '--data_root',
        type=str, default='/home/tanimu/data',
        help='root path to cifar10-c directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int, default=1024,
        help='batch size',
    )

    parser.add_argument(
        '--corruptions',
        type=str, nargs='*',
        default=corruptions,
        help='testing corruption types',
    )
    parser.add_argument(
        '--levels',
        type=int, nargs='*',
        choices=[1, 2, 3, 4, 5],
        default=[1, 2, 3, 4, 5],
        help='testing corruption levels',
    )

    parser.add_argument(
        '--gpu_id',
        type=str, default=0,
        help='gpu id to use'
    )

    opt = parser.parse_args()
    main(opt)