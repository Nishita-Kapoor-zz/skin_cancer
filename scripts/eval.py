from torch.autograd import Variable
from utils import plot_confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from data.dataloader import *
import matplotlib.pyplot as plt


def evaluate(args, model, device):
    _, dataloaders = create_dataloaders(args)
    model.eval()
    y_label = []
    y_predict = []
    with torch.no_grad():
        for i, data in enumerate(dataloaders['test']):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]
            y_label.extend(labels.cpu().numpy())
            y_predict.extend(np.squeeze(prediction.cpu().numpy().T))

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(y_label, y_predict)
    # plot the confusion matrix
    plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc', 'mel']

    plot_confusion_matrix(confusion_mtx, plot_labels)

    report = classification_report(y_label, y_predict, target_names=plot_labels)
    print(report)

    #label_frac_error = 1 - np.diag(confusion_mtx) / np.sum(confusion_mtx, axis=1)
    #plt.bar(np.arange(7), label_frac_error)
    #plt.xlabel('True Label')
    #plt.ylabel('Fraction classified incorrectly')