# main.py COPYRIGHT Fujitsu Limited 2023

from argparse import ArgumentParser
from collections import OrderedDict
import copy
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
from tqdm import tqdm

import sys
sys.path.append('../../')
from auto_prune_cifar10 import auto_prune

from resnet110 import ResNet110

#===================================================================================
parser = ArgumentParser()
parser.add_argument('--workers', default=8, type=int,
                    help='number of data loading workers')
parser.add_argument('--use_gpu', action='store_true',
                    help='use gpu')
parser.add_argument('--use_DataParallel', action='store_true',
                    help='use DataParallel')
# for training
parser.add_argument('--data', type=str, default='./data',
                    help='path to dataset')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--learning_rate', type=float, default=1e-1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--nesterov', default=False)
parser.add_argument('--scheduler_timing', type=str, default='epoch',
                    help="set LR change timing by LR_scheduler. 'epoch': execute scheduler.step() for each epoch. 'iter' : Execute scheduler.step() for each iteration")
# for stepLR scheduler
parser.add_argument('--lr-milestone', type=list, default=[100, 150, 200])
parser.add_argument('--lr-gamma', type=float, default=0.1)
# for auto pruning
parser.add_argument('--acc_control', type=float, default=0.1,
                    help='control parameter for pruned model accuracy')
parser.add_argument('--loss_margin', type=float, default=0.1,
                    help='control parameter for loss function margin to derive threshold')
parser.add_argument('--rates', nargs='*', type=float, default=[0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
                    help='candidates for pruning rates')
parser.add_argument('--max_search_times', type=int, default=1000,
                    help='maximum number of times for pruning rate search')
parser.add_argument('--epochs', type=int, default=250,
                    help='re-training epochs')
parser.add_argument('--pretrained_model_path', type=str, default='./pretrained_cifar10_resnet110.pt',
                    help='pre-trained model filepath')
parser.add_argument('--pruned_model_path', type=str, default='./pruned_cifar10_resnet110.pt',
                    help='pruned model filepath')
#===================================================================================

def main():
    args = parser.parse_args()
    args.rates = ([float(f) for f in args.rates])
    print(f'args: {args}')

    device = 'cpu'
    if args.use_gpu:
        torch.backends.cudnn.benchmark = True
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('device: ', device)

    print('===== load data ===================')
    norm_mean = (0.485, 0.456, 0.406)
    norm_std  = (0.229, 0.224, 0.225)
    train_transform = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )

    val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(norm_mean, norm_std),
            ]
        )

    # get cifar10 datasets
    dataset_path = args.data
    train_dataset = datasets.CIFAR10(
        root=dataset_path, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(
        root=dataset_path, train=False, download=True, transform=val_transform)

    # make DataLoader
    batch_size = args.batch_size
    train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=True,
        )
    val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=args.workers,
            pin_memory=True,
        )

    # load model
    model = ResNet110()
    model.load_state_dict(torch.load(
        args.pretrained_model_path, map_location=device), strict=True)

    if torch.cuda.device_count() > 1 and args.use_DataParallel:
        print('use {} GPUs.'.format(torch.cuda.device_count()))
        model = torch.nn.DataParallel(model)
    model.to(device)
    print('===== model: before pruning ==========')
    print(model)
    # Model information for pruning
    model_info = OrderedDict()
    model_info['conv1'] = {'arg': 'ch_conv1'}
    model_info['bn1']   = {'arg': 'ch_conv1'}

    model_info['l10_conv1'] = {'arg': 'ch_l10conv1', 'prev': ['bn1']}
    model_info['l10_bn1']   = {'arg': 'ch_l10conv1'}
    model_info['l10_conv2'] = {'arg': 'ch_l10conv2'}
    model_info['l10_bn2']   = {'arg': 'ch_l10conv2'}

    model_info['l11_conv1'] = {'arg': 'ch_l11conv1', 'prev': ['bn1', 'l10_bn2']}
    model_info['l11_bn1']   = {'arg': 'ch_l11conv1'}
    model_info['l11_conv2'] = {'arg': 'ch_l11conv2'}
    model_info['l11_bn2']   = {'arg': 'ch_l11conv2'}

    model_info['l12_conv1'] = {'arg': 'ch_l12conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2']}
    model_info['l12_bn1']   = {'arg': 'ch_l12conv1'}
    model_info['l12_conv2'] = {'arg': 'ch_l12conv2'}
    model_info['l12_bn2']   = {'arg': 'ch_l12conv2'}

    model_info['l13_conv1'] = {'arg': 'ch_l13conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2']}
    model_info['l13_bn1']   = {'arg': 'ch_l13conv1'}
    model_info['l13_conv2'] = {'arg': 'ch_l13conv2'}
    model_info['l13_bn2']   = {'arg': 'ch_l13conv2'}

    model_info['l14_conv1'] = {'arg': 'ch_l14conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2']}
    model_info['l14_bn1']   = {'arg': 'ch_l14conv1'}
    model_info['l14_conv2'] = {'arg': 'ch_l14conv2'}
    model_info['l14_bn2']   = {'arg': 'ch_l14conv2'}

    model_info['l15_conv1'] = {'arg': 'ch_l15conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2']}
    model_info['l15_bn1']   = {'arg': 'ch_l15conv1'}
    model_info['l15_conv2'] = {'arg': 'ch_l15conv2'}
    model_info['l15_bn2']   = {'arg': 'ch_l15conv2'}

    model_info['l16_conv1'] = {'arg': 'ch_l16conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2']}
    model_info['l16_bn1']   = {'arg': 'ch_l16conv1'}
    model_info['l16_conv2'] = {'arg': 'ch_l16conv2'}
    model_info['l16_bn2']   = {'arg': 'ch_l16conv2'}

    model_info['l17_conv1'] = {'arg': 'ch_l17conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2']}
    model_info['l17_bn1']   = {'arg': 'ch_l17conv1'}
    model_info['l17_conv2'] = {'arg': 'ch_l17conv2'}
    model_info['l17_bn2']   = {'arg': 'ch_l17conv2'}

    model_info['l18_conv1'] = {'arg': 'ch_l18conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2']}
    model_info['l18_bn1']   = {'arg': 'ch_l18conv1'}
    model_info['l18_conv2'] = {'arg': 'ch_l18conv2'}
    model_info['l18_bn2']   = {'arg': 'ch_l18conv2'}

    model_info['l19_conv1'] = {'arg': 'ch_l19conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2','l18_bn2']}

    model_info['l19_bn1']   = {'arg': 'ch_l19conv1'}
    model_info['l19_conv2'] = {'arg': 'ch_l19conv2'}
    model_info['l19_bn2']   = {'arg': 'ch_l19conv2'}

    model_info['l110_conv1'] = {'arg': 'ch_l110conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2','l18_bn2',
                                                                'l19_bn2']}
    model_info['l110_bn1']   = {'arg': 'ch_l110conv1'}
    model_info['l110_conv2'] = {'arg': 'ch_l110conv2'}
    model_info['l110_bn2']   = {'arg': 'ch_l110conv2'}

    model_info['l111_conv1'] = {'arg': 'ch_l111conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2','l18_bn2',
                                                                'l19_bn2','l110_bn2']}
    model_info['l111_bn1']   = {'arg': 'ch_l111conv1'}
    model_info['l111_conv2'] = {'arg': 'ch_l111conv2'}
    model_info['l111_bn2']   = {'arg': 'ch_l111conv2'}

    model_info['l112_conv1'] = {'arg': 'ch_l112conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2','l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2']}
    model_info['l112_bn1']   = {'arg': 'ch_l112conv1'}
    model_info['l112_conv2'] = {'arg': 'ch_l112conv2'}
    model_info['l112_bn2']   = {'arg': 'ch_l112conv2'}

    model_info['l113_conv1'] = {'arg': 'ch_l113conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2','l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2']}
    model_info['l113_bn1']   = {'arg': 'ch_l113conv1'}
    model_info['l113_conv2'] = {'arg': 'ch_l113conv2'}
    model_info['l113_bn2']   = {'arg': 'ch_l113conv2'}

    model_info['l114_conv1'] = {'arg': 'ch_l114conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2','l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2']}
    model_info['l114_bn1']   = {'arg': 'ch_l114conv1'}
    model_info['l114_conv2'] = {'arg': 'ch_l114conv2'}
    model_info['l114_bn2']   = {'arg': 'ch_l114conv2'}

    model_info['l115_conv1'] = {'arg': 'ch_l115conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2','l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2']}
    model_info['l115_bn1']   = {'arg': 'ch_l115conv1'}
    model_info['l115_conv2'] = {'arg': 'ch_l115conv2'}
    model_info['l115_bn2']   = {'arg': 'ch_l115conv2'}

    model_info['l116_conv1'] = {'arg': 'ch_l116conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2','l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2']}
    model_info['l116_bn1']   = {'arg': 'ch_l116conv1'}
    model_info['l116_conv2'] = {'arg': 'ch_l116conv2'}
    model_info['l116_bn2']   = {'arg': 'ch_l116conv2'}

    model_info['l117_conv1'] = {'arg': 'ch_l117conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2','l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2']}
    model_info['l117_bn1']   = {'arg': 'ch_l117conv1'}
    model_info['l117_conv2'] = {'arg': 'ch_l117conv2'}
    model_info['l117_bn2']   = {'arg': 'ch_l117conv2'}


    model_info['l20_conv1'] = {'arg': 'ch_l20conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2']}
    model_info['l20_bn1']   = {'arg': 'ch_l20conv1'}
    model_info['l20_conv2'] = {'arg': 'ch_l20conv2'}
    model_info['l20_bn2']   = {'arg': 'ch_l20conv2'}

    model_info['l21_conv1'] = {'arg': 'ch_l21conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2']}
    model_info['l21_bn1']   = {'arg': 'ch_l21conv1'}
    model_info['l21_conv2'] = {'arg': 'ch_l21conv2'}
    model_info['l21_bn2']   = {'arg': 'ch_l21conv2'}

    model_info['l22_conv1'] = {'arg': 'ch_l22conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2']}
    model_info['l22_bn1']   = {'arg': 'ch_l22conv1'}
    model_info['l22_conv2'] = {'arg': 'ch_l22conv2'}
    model_info['l22_bn2']   = {'arg': 'ch_l22conv2'}

    model_info['l23_conv1'] = {'arg': 'ch_l23conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2']}
    model_info['l23_bn1']   = {'arg': 'ch_l23conv1'}
    model_info['l23_conv2'] = {'arg': 'ch_l23conv2'}
    model_info['l23_bn2']   = {'arg': 'ch_l23conv2'}

    model_info['l24_conv1'] = {'arg': 'ch_l24conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2']}
    model_info['l24_bn1']   = {'arg': 'ch_l24conv1'}
    model_info['l24_conv2'] = {'arg': 'ch_l24conv2'}
    model_info['l24_bn2']   = {'arg': 'ch_l24conv2'}

    model_info['l25_conv1'] = {'arg': 'ch_l25conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2']}
    model_info['l25_bn1']   = {'arg': 'ch_l25conv1'}
    model_info['l25_conv2'] = {'arg': 'ch_l25conv2'}
    model_info['l25_bn2']   = {'arg': 'ch_l25conv2'}

    model_info['l26_conv1'] = {'arg': 'ch_l26conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2']}
    model_info['l26_bn1']   = {'arg': 'ch_l26conv1'}
    model_info['l26_conv2'] = {'arg': 'ch_l26conv2'}
    model_info['l26_bn2']   = {'arg': 'ch_l26conv2'}

    model_info['l27_conv1'] = {'arg': 'ch_l27conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2']}
    model_info['l27_bn1']   = {'arg': 'ch_l27conv1'}
    model_info['l27_conv2'] = {'arg': 'ch_l27conv2'}
    model_info['l27_bn2']   = {'arg': 'ch_l27conv2'}

    model_info['l28_conv1'] = {'arg': 'ch_l28conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2']}
    model_info['l28_bn1']   = {'arg': 'ch_l28conv1'}
    model_info['l28_conv2'] = {'arg': 'ch_l28conv2'}
    model_info['l28_bn2']   = {'arg': 'ch_l28conv2'}

    model_info['l29_conv1'] = {'arg': 'ch_l29conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2','l28_bn2']}

    model_info['l29_bn1']   = {'arg': 'ch_l29conv1'}
    model_info['l29_conv2'] = {'arg': 'ch_l29conv2'}
    model_info['l29_bn2']   = {'arg': 'ch_l29conv2'}

    model_info['l210_conv1'] = {'arg': 'ch_l210conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2','l28_bn2',
                                                                'l29_bn2']}
    model_info['l210_bn1']   = {'arg': 'ch_l210conv1'}
    model_info['l210_conv2'] = {'arg': 'ch_l210conv2'}
    model_info['l210_bn2']   = {'arg': 'ch_l210conv2'}

    model_info['l211_conv1'] = {'arg': 'ch_l211conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2','l28_bn2',
                                                                'l29_bn2','l210_bn2']}
    model_info['l211_bn1']   = {'arg': 'ch_l211conv1'}
    model_info['l211_conv2'] = {'arg': 'ch_l211conv2'}
    model_info['l211_bn2']   = {'arg': 'ch_l211conv2'}

    model_info['l212_conv1'] = {'arg': 'ch_l212conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2','l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2']}
    model_info['l212_bn1']   = {'arg': 'ch_l212conv1'}
    model_info['l212_conv2'] = {'arg': 'ch_l212conv2'}
    model_info['l212_bn2']   = {'arg': 'ch_l212conv2'}

    model_info['l213_conv1'] = {'arg': 'ch_l213conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2','l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2','l212_bn2']}
    model_info['l213_bn1']   = {'arg': 'ch_l213conv1'}
    model_info['l213_conv2'] = {'arg': 'ch_l213conv2'}
    model_info['l213_bn2']   = {'arg': 'ch_l213conv2'}

    model_info['l214_conv1'] = {'arg': 'ch_l214conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2','l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2']}
    model_info['l214_bn1']   = {'arg': 'ch_l214conv1'}
    model_info['l214_conv2'] = {'arg': 'ch_l214conv2'}
    model_info['l214_bn2']   = {'arg': 'ch_l214conv2'}

    model_info['l215_conv1'] = {'arg': 'ch_l215conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2','l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2']}
    model_info['l215_bn1']   = {'arg': 'ch_l215conv1'}
    model_info['l215_conv2'] = {'arg': 'ch_l215conv2'}
    model_info['l215_bn2']   = {'arg': 'ch_l215conv2'}

    model_info['l216_conv1'] = {'arg': 'ch_l216conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2','l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2']}
    model_info['l216_bn1']   = {'arg': 'ch_l216conv1'}
    model_info['l216_conv2'] = {'arg': 'ch_l216conv2'}
    model_info['l216_bn2']   = {'arg': 'ch_l216conv2'}

    model_info['l217_conv1'] = {'arg': 'ch_l217conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2','l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2']}
    model_info['l217_bn1']   = {'arg': 'ch_l217conv1'}
    model_info['l217_conv2'] = {'arg': 'ch_l217conv2'}
    model_info['l217_bn2']   = {'arg': 'ch_l217conv2'}


    model_info['l30_conv1'] = {'arg': 'ch_l30conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                              'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2']}

    model_info['l30_bn1']   = {'arg': 'ch_l30conv1'}
    model_info['l30_conv2'] = {'arg': 'ch_l30conv2'}
    model_info['l30_bn2']   = {'arg': 'ch_l30conv2'}

    model_info['l31_conv1'] = {'arg': 'ch_l31conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                              'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                              'l30_bn2']}
    model_info['l31_bn1']   = {'arg': 'ch_l31conv1'}
    model_info['l31_conv2'] = {'arg': 'ch_l31conv2'}
    model_info['l31_bn2']   = {'arg': 'ch_l31conv2'}

    model_info['l32_conv1'] = {'arg': 'ch_l32conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                              'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                              'l30_bn2', 'l31_bn2']}
    model_info['l32_bn1']   = {'arg': 'ch_l32conv1'}
    model_info['l32_conv2'] = {'arg': 'ch_l32conv2'}
    model_info['l32_bn2']   = {'arg': 'ch_l32conv2'}

    model_info['l33_conv1'] = {'arg': 'ch_l33conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                              'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                              'l30_bn2', 'l31_bn2', 'l32_bn2']}
    model_info['l33_bn1']   = {'arg': 'ch_l33conv1'}
    model_info['l33_conv2'] = {'arg': 'ch_l33conv2'}
    model_info['l33_bn2']   = {'arg': 'ch_l33conv2'}

    model_info['l34_conv1'] = {'arg': 'ch_l34conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                              'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                              'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2']}
    model_info['l34_bn1']   = {'arg': 'ch_l34conv1'}
    model_info['l34_conv2'] = {'arg': 'ch_l34conv2'}
    model_info['l34_bn2']   = {'arg': 'ch_l34conv2'}

    model_info['l35_conv1'] = {'arg': 'ch_l35conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                              'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                              'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2']}
    model_info['l35_bn1']   = {'arg': 'ch_l35conv1'}
    model_info['l35_conv2'] = {'arg': 'ch_l35conv2'}
    model_info['l35_bn2']   = {'arg': 'ch_l35conv2'}

    model_info['l36_conv1'] = {'arg': 'ch_l36conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                              'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                              'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2', 'l35_bn2']}
    model_info['l36_bn1']   = {'arg': 'ch_l36conv1'}
    model_info['l36_conv2'] = {'arg': 'ch_l36conv2'}
    model_info['l36_bn2']   = {'arg': 'ch_l36conv2'}

    model_info['l37_conv1'] = {'arg': 'ch_l37conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                              'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                              'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2', 'l35_bn2', 'l36_bn2']}
    model_info['l37_bn1']   = {'arg': 'ch_l37conv1'}
    model_info['l37_conv2'] = {'arg': 'ch_l37conv2'}
    model_info['l37_bn2']   = {'arg': 'ch_l37conv2'}

    model_info['l38_conv1'] = {'arg': 'ch_l38conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                              'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                              'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2', 'l35_bn2', 'l36_bn2', 'l37_bn2']}
    model_info['l38_bn1']   = {'arg': 'ch_l38conv1'}
    model_info['l38_conv2'] = {'arg': 'ch_l38conv2'}
    model_info['l38_bn2']   = {'arg': 'ch_l38conv2'}

    model_info['l39_conv1'] = {'arg': 'ch_l39conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                              'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                              'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                              'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                              'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2', 'l35_bn2', 'l36_bn2', 'l37_bn2','l38_bn2']}

    model_info['l39_bn1']   = {'arg': 'ch_l39conv1'}
    model_info['l39_conv2'] = {'arg': 'ch_l39conv2'}
    model_info['l39_bn2']   = {'arg': 'ch_l39conv2'}

    model_info['l310_conv1'] = {'arg': 'ch_l310conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                                'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2', 'l35_bn2', 'l36_bn2', 'l37_bn2','l38_bn2',
                                                                'l39_bn2']}
    model_info['l310_bn1']   = {'arg': 'ch_l310conv1'}
    model_info['l310_conv2'] = {'arg': 'ch_l310conv2'}
    model_info['l310_bn2']   = {'arg': 'ch_l310conv2'}

    model_info['l311_conv1'] = {'arg': 'ch_l311conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                                'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2', 'l35_bn2', 'l36_bn2', 'l37_bn2','l38_bn2',
                                                                'l39_bn2','l310_bn2']}
    model_info['l311_bn1']   = {'arg': 'ch_l311conv1'}
    model_info['l311_conv2'] = {'arg': 'ch_l311conv2'}
    model_info['l311_bn2']   = {'arg': 'ch_l311conv2'}

    model_info['l312_conv1'] = {'arg': 'ch_l312conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                                'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2', 'l35_bn2', 'l36_bn2', 'l37_bn2','l38_bn2',
                                                                'l39_bn2','l310_bn2','l311_bn2']}
    model_info['l312_bn1']   = {'arg': 'ch_l312conv1'}
    model_info['l312_conv2'] = {'arg': 'ch_l312conv2'}
    model_info['l312_bn2']   = {'arg': 'ch_l312conv2'}

    model_info['l313_conv1'] = {'arg': 'ch_l313conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                                'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2', 'l35_bn2', 'l36_bn2', 'l37_bn2','l38_bn2',
                                                                'l39_bn2','l310_bn2','l311_bn2','l312_bn2']}
    model_info['l313_bn1']   = {'arg': 'ch_l313conv1'}
    model_info['l313_conv2'] = {'arg': 'ch_l313conv2'}
    model_info['l313_bn2']   = {'arg': 'ch_l313conv2'}

    model_info['l314_conv1'] = {'arg': 'ch_l314conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                                'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2', 'l35_bn2', 'l36_bn2', 'l37_bn2','l38_bn2',
                                                                'l39_bn2','l310_bn2','l311_bn2','l312_bn2','l313_bn2']}
    model_info['l314_bn1']   = {'arg': 'ch_l314conv1'}
    model_info['l314_conv2'] = {'arg': 'ch_l314conv2'}
    model_info['l314_bn2']   = {'arg': 'ch_l314conv2'}

    model_info['l315_conv1'] = {'arg': 'ch_l315conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                                'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2', 'l35_bn2', 'l36_bn2', 'l37_bn2','l38_bn2',
                                                                'l39_bn2','l310_bn2','l311_bn2','l312_bn2','l313_bn2','l314_bn2']}
    model_info['l315_bn1']   = {'arg': 'ch_l315conv1'}
    model_info['l315_conv2'] = {'arg': 'ch_l315conv2'}
    model_info['l315_bn2']   = {'arg': 'ch_l315conv2'}

    model_info['l316_conv1'] = {'arg': 'ch_l316conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                                'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2', 'l35_bn2', 'l36_bn2', 'l37_bn2','l38_bn2',
                                                                'l39_bn2','l310_bn2','l311_bn2','l312_bn2','l313_bn2','l314_bn2','l315_bn2']}
    model_info['l316_bn1']   = {'arg': 'ch_l316conv1'}
    model_info['l316_conv2'] = {'arg': 'ch_l316conv2'}
    model_info['l316_bn2']   = {'arg': 'ch_l316conv2'}

    model_info['l317_conv1'] = {'arg': 'ch_l317conv1', 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                                'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                                'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                                'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                                'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2', 'l35_bn2', 'l36_bn2', 'l37_bn2','l38_bn2',
                                                                'l39_bn2','l310_bn2','l311_bn2','l312_bn2','l313_bn2','l314_bn2','l315_bn2','l316_bn2']}
    model_info['l317_bn1']   = {'arg': 'ch_l317conv1'}
    model_info['l317_conv2'] = {'arg': 'ch_l317conv2'}
    model_info['l317_bn2']   = {'arg': 'ch_l317conv2'}

    model_info['linear'] = {'arg': None, 'prev': ['bn1', 'l10_bn2', 'l11_bn2', 'l12_bn2', 'l13_bn2', 'l14_bn2', 'l15_bn2', 'l16_bn2', 'l17_bn2', 'l18_bn2',
                                                  'l19_bn2','l110_bn2','l111_bn2','l112_bn2','l113_bn2','l114_bn2','l115_bn2','l116_bn2','l117_bn2',
                                                  'l20_bn2', 'l21_bn2', 'l22_bn2', 'l23_bn2', 'l24_bn2', 'l25_bn2', 'l26_bn2', 'l27_bn2', 'l28_bn2',
                                                  'l29_bn2','l210_bn2','l211_bn2','l212_bn2','l213_bn2','l214_bn2','l215_bn2','l216_bn2', 'l217_bn2',
                                                  'l30_bn2', 'l31_bn2', 'l32_bn2', 'l33_bn2', 'l34_bn2', 'l35_bn2', 'l36_bn2', 'l37_bn2', 'l38_bn2', 'l38_bn2',
                                                  'l39_bn2','l310_bn2','l311_bn2','l312_bn2','l313_bn2','l314_bn2','l315_bn2','l316_bn2', 'l317_bn2']}


    # load weight of trained model
    if torch.cuda.device_count() > 1 and args.use_DataParallel:
        weights = copy.deepcopy(model.module.state_dict())
    else:
        weights = copy.deepcopy(model.state_dict())

    # calculate accuracy with unpruned trained model
    Ab = validate(val_loader, model, device, epoch=1)
    print('Accuracy before pruning:', Ab)

    # tune pruning rate
    print('===== start pruning rate tuning =====')
    ### set optimizer
    optim_params = dict(lr=args.learning_rate,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        nesterov=args.nesterov)
    ### set LR scheduler
    # step LR scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR
    scheduler_params = dict(milestones=args.lr_milestone, gamma=args.lr_gamma)

    criterion = torch.nn.CrossEntropyLoss()

    weights, Afinal, n_args_channels = auto_prune(ResNet110, model_info, weights, Ab,
                                                  train_loader, val_loader, criterion,
                                                  optim_type='SGD',
                                                  optim_params=optim_params,
                                                  lr_scheduler=scheduler,
                                                  scheduler_params=scheduler_params,
                                                  update_lr=args.scheduler_timing,
                                                  use_gpu=args.use_gpu,
                                                  use_DataParallel=args.use_DataParallel,
                                                  acc_control=args.acc_control,
                                                  loss_margin=args.loss_margin,
                                                  rates=args.rates,
                                                  max_search_times=args.max_search_times,
                                                  epochs=args.epochs,
                                                  model_path=args.pretrained_model_path,
                                                  pruned_model_path=args.pruned_model_path)
                                                  
    print('===== model: after pruning ==========')
    print(model)
    print('===== Results =====')
    print('Model size before pruning (Byte):', os.path.getsize(args.pretrained_model_path))
    if os.path.exists(args.pruned_model_path):
        print('Model size after pruning  (Byte):',
              os.path.getsize(args.pruned_model_path))
        print('Compression rate                : {:.3f}'.format(
            1-os.path.getsize(args.pruned_model_path)/os.path.getsize(args.pretrained_model_path)))
    print('Acc. before pruning: {:.2f}'.format(Ab))
    print('Acc. after pruning : {:.2f}'.format(Afinal))
    print('Arguments name & number of channels for pruned model: ', n_args_channels)


def validate(val_loader, model, device, epoch):
    model.eval()
    with torch.no_grad():
        with tqdm(val_loader, leave=False) as pbar:
            pbar.set_description('Epoch {} Validation'.format(epoch))
            hit = 0
            total = 0
            for _, (images, targets) in enumerate(pbar):
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                outClass = outputs.cpu().detach().numpy().argmax(axis=1)
                hit += (outClass == targets.cpu().numpy()).sum()
                total += len(targets)
                val_acc = hit / total * 100
                pbar.set_postfix({'valid Acc': val_acc})
    return val_acc

if __name__ == '__main__':
    main()

