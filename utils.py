from torch.utils.data import Dataset
import numpy as np
from PIL import Image
from tqdm import tqdm
from collections import OrderedDict
import cv2
import matplotlib.pyplot as plt
from torch import nn
import os
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import fbeta_score


class CustomDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)
        return X, y


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _one_hot(index, classes):
    device = index.device
    size = index.size() + (classes,)
    view = index.size() + (1,)

    mask = torch.Tensor(*size).fill_(0)
    index = index.view(*view)
    ones = 1.0

    if isinstance(index, Variable):
        ones = Variable(torch.Tensor(index.size()).fill_(1))
        mask = Variable(mask, requires_grad=index.requires_grad)

    return mask.to(device).scatter_(1, index, ones.to(device))


class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        if isinstance(self.weight, torch.Tensor):
            self.weight = torch.ones(input.size(-1))
        assert self.weight.size(0) == input.size(-1)
        device = input.device
        self.weight = self.weight.to(device)

        y = _one_hot(target, input.size(-1))
        logit = F.softmax(input, dim=-1)
        logit = logit.clamp(self.eps, 1.0 - self.eps)

        loss = -1 * y * torch.log(logit)
        loss = self.weight * loss * (1 - logit) ** self.gamma

        return loss.sum()


def compute_img_mean_std(image_paths):
    """
        computing the mean and std of three channels on the whole dataset,
        first we should normalize the image from 0-255 to 0-1
    """

    img_h, img_w = 224, 224
    imgs = []
    means, stdevs = [], []

    for i in tqdm(range(len(image_paths))):
        img = cv2.imread(image_paths[i])
        img = cv2.resize(img, (img_h, img_w))
        imgs.append(img)

    imgs = np.stack(imgs, axis=3)
    print(imgs.shape)

    imgs = imgs.astype(np.float32) / 255.

    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()  # resize to one row
        means.append(np.mean(pixels))
        stdevs.append(np.std(pixels))

    means.reverse()  # BGR --> RGB
    stdevs.reverse()

    print("normMean = {}".format(means))
    print("normStd = {}".format(stdevs))
    return means, stdevs


def plot_confusion_matrix(cm, fig_path, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    import itertools
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(fig_path)


def remove_redundant_keys(state_dict: OrderedDict):
    # remove DataParallel wrapping
    if "module" in list(state_dict.keys())[0]:
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                new_state_dict[k[7:]] = v
    else:
        new_state_dict = state_dict

    return new_state_dict


def save_checkpoint(path, model, epoch, optimizer=None, save_arch=False, params=None):
    attributes = {"epoch": epoch, "state_dict": remove_redundant_keys(model.state_dict())}
    if optimizer is not None:
        attributes["optimizer"] = optimizer.state_dict()
    if save_arch:
        attributes["arch"] = model
    if params is not None:
        attributes["params"] = params

    try:
        torch.save(attributes, path)
    except TypeError:
        if "arch" in attributes:
            print(
                "Model architecture will be ignored because the architecture includes non-pickable objects."
            )
            del attributes["arch"]
            torch.save(attributes, path)


def load_checkpoint(path, model, optimizer=None, params=False, epoch=False):
    resume = torch.load(path)
    rets = dict()

    if isinstance(model, nn.DataParallel):
        model.module.load_state_dict(remove_redundant_keys(resume["state_dict"]))
    else:
        model.load_state_dict(remove_redundant_keys(resume["state_dict"]))

        rets["model"] = model

    if optimizer is not None:
        optimizer.load_state_dict(resume["optimizer"])
        rets["optimizer"] = optimizer
    if params:
        rets["params"] = resume["params"]
    if epoch:
        rets["epoch"] = resume["epoch"]

    return rets


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


def avg_fscore(y_true, y_pred):
    return fbeta_score(y_true, y_pred, average="macro", beta=0.5)

