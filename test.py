import argparse
import os
import pprint
import torch
import torchvision
import tqdm

import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from utils import load_txt, accuracy
from models.resnet import ResNet56
from dataset import CIFAR10C, extract_subset

corruptions = load_txt('./corruptions.txt')


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

    accs = dict()
    with tqdm(total=len(opt.corruptions)*len(opt.levels), ncols=80) as pbar:
        for cname in opt.corruptions:
            for level in opt.levels:
                dataset = CIFAR10C(opt.data_root, cname, level,
                                   transform=transforms.ToTensor())
                if opt.num_samples is not None:
                    dataset = extract_subset(dataset, opt.num_samples, False)
                loader = DataLoader(dataset, batch_size=opt.batch_size,
                                    shuffle=False, num_workers=4)
                
                with torch.no_grad():
                    for itr, (x, y) in enumerate(loader):
                        x = x.to(device, non_blocking=True)
                        y = y.to(device, dtype=torch.int64, non_blocking=True)

                        # calcurate clean loss and accuracy
                        z = model(x)
                        loss = F.cross_entropy(z, y)
                        acc, _ = accuracy(z, y, topk=(1, 5))

                        # report the training status
                        pbar.set_postfix_str(f'{cname}-{level}: {acc.item():.2f}')
                        pbar.update()
                
                accs[f'{cname}-{level}'] = acc.item()
    
    pprint.pprint(accs)


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
        type=str, default='/home/tanimu/data/cifar10-c',
        help='root path to cifar10-c directory'
    )
    parser.add_argument(
        '--batch_size',
        type=int, default=1024,
        help='batch size',
    )
    parser.add_argument(
        '--num_samples',
        type=int, default=1000,
        help='number of samples to test',
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