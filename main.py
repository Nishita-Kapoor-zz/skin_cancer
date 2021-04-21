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
from scripts.eval import evaluate, predict


parser = argparse.ArgumentParser(description="PyTorch classification of Skin Cancer MNIST")
parser.add_argument('--view_data', type=bool, default=False, help='Visualize data distribution')
parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs to train on')
parser.add_argument('--train', default=True, type=bool, help='Train the model')
parser.add_argument('--test', default=True, type=bool, help='Test the model')
parser.add_argument('--gpus', default="0", type=str, help='Which GPU to use?')
parser.add_argument('--path', default='/home/nishita/datasets/skin_mnist', type=str, help='Path of dataset')
parser.add_argument("--version", "-v", default=1, type=str, help="Version of experiment/run")
parser.add_argument("--batch_size", default=32, type=int, help="batch-size to use")
parser.add_argument("--lr", default=1e-4, type=float, help="learning rate to use")
parser.add_argument("--image_path", default=None, type=str, help="Path for single image prediction (inference)")


args = parser.parse_args()
print(f'The arguments are {vars(args)}')


if torch.cuda.is_available():
    os.environ["CUDA_VISION_DEVICES"] = str(args.gpus)
    device = torch.device("cuda:" + str(args.gpus))
    print("CUDA GPU {} found.".format(args.gpus))
else:
    device = torch.device("cpu")
    print("No CUDA device found. Using CPU instead.")


model = models.resnext101_32x8d(pretrained=True)
model.fc = nn.Linear(in_features=2048, out_features=7)

model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
# criterion = FocalLoss(weight=weights, gamma=2).to(device)
criterion = nn.CrossEntropyLoss().to(device)

if args.train:
    training(args, model, criterion, optimizer, device)

if args.test:
    evaluate(args, device, model)

if args.image_path is not None:
    predict(args, device=device, model=model)

if args.view_data:
    view_samples(args)
    data_dist(args)

