"""
Image classification of skin lesions from HAM10000 dataset using pre-trained DenseNet

Written by: Nishita Kapoor
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from data.visualize import *
from torchvision import models
from scripts.train import training
from scripts.eval import evaluate


parser = argparse.ArgumentParser(description="PyTorch classification of Skin Cancer MNIST")
parser.add_argument('--view_data', type=bool, default=False, help='Visualize data distribution')
parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs to train on')
parser.add_argument('--train', default=True, type=bool, help='Train the model')
parser.add_argument('--test', default=False, type=bool, help='Train the model')
parser.add_argument("--checkpoint", default=None, type=str, help="Path of checkpoint to evaluate from")
parser.add_argument('--path', default='/home/nishita/datasets/skin_mnist', type=str, help='Path of dataset')
parser.add_argument("--experiment", "-exp", default=1, type=int, help="Version of experiment")
parser.add_argument("--batch_size", default=2, type=int, help="batch-size to use")
parser.add_argument("--lr", default=1e-3, type=float, help="learning rate to use")
parser.add_argument("--model", default="resnet18", help="network architecture")

args = parser.parse_args()
print(f'The arguments are {vars(args)}')


if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print("CUDA GPU found.")
else:
    device = torch.device("cpu")
    print("No CUDA device found. Using CPU instead.")


model = models.resnext101_32x8d(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=7)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
criterion = nn.CrossEntropyLoss().to(device)

if args.train:
    training(args, model, criterion, optimizer, device)

if args.test:
    if args.checkpoint is not None:
        evaluate(args, model, criterion, optimizer, device)  ##fix!!
    else:
        raise AssertionError("Checkpoint path not found")

if args.view_data:
    view_samples(args)
    data_dist(args)
