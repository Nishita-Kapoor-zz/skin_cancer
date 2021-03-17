"""
Image classification of skin lesions from HAM10000 dataset using pre-trained DenseNet

Written by: Nishita Kapoor
"""
import argparse
import torch
from glob import glob
import os
from data.visualize import *
from torch.utils.tensorboard import SummaryWriter
import logging

device = torch.device("cpu")

parser = argparse.ArgumentParser(description="PyTorch classification of Skin Cancer MNIST")
parser.add_argument('--use_cuda', type=bool, default=True, help='Device to train on')
parser.add_argument('--view_data', type=bool, default=True, help='Visualize data distribution')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train on')
parser.add_argument('--train', default=True, type=bool, help='Train the model')
parser.add_argument('--path', default='/home/nishita/datasets/skin_mnist', type=str, help='Path of dataset')
parser.add_argument("--version", default=1, type=int, help="Version of experiment")
parser.add_argument("--batch-size", default=8, type=int, help="batch-size to use")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate to use")
parser.add_argument("--model", default="resnet18", help="network architecture")

args = parser.parse_args()
print(f'The arguments are {vars(args)}')

if args.use_cuda:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

base_dir = args.path
all_image_path = glob(os.path.join(base_dir, '*', '*.jpg'))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}

if args.view_data:
    view_samples(imageid_path_dict, all_image_path)
    data_dist(args)