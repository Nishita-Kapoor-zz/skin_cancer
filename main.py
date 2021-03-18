"""
Image classification of skin lesions from HAM10000 dataset using pre-trained DenseNet

Written by: Nishita Kapoor
"""
import argparse
import torch
from glob import glob
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from data.visualize import *
from torchvision import models
from torch.utils.tensorboard import SummaryWriter
import logging
from scripts.train import training

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(device)
parser = argparse.ArgumentParser(description="PyTorch classification of Skin Cancer MNIST")
parser.add_argument('--use_cuda', type=bool, default=True, help='Device to train on')
parser.add_argument('--view_data', type=bool, default=False, help='Visualize data distribution')
parser.add_argument('--num_epochs', type=int, default=1, help='Number of epochs to train on')
parser.add_argument('--train', default=True, type=bool, help='Train the model')
parser.add_argument('--path', default='/home/nishita/datasets/skin_mnist', type=str, help='Path of dataset')
parser.add_argument("--version", default=1, type=int, help="Version of experiment")
parser.add_argument("--batch_size", default=2, type=int, help="batch-size to use")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate to use")
parser.add_argument("--model", default="resnet18", help="network architecture")

args = parser.parse_args()
print(f'The arguments are {vars(args)}')

#if args.use_cuda:
#    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#    print(device)
model = models.resnext101_32x8d(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=7)

#model.to(device)

model.to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss().to(device)

if args.train:
    training(args, model, criterion, optimizer, device)

if args.view_data:
    view_samples(args)
    data_dist(args)
