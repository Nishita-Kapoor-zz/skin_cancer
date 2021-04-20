from torch.autograd import Variable
from tqdm import tqdm
from utils import AverageMeter
from torch.utils.tensorboard import SummaryWriter

from data.dataloader import *
import matplotlib.pyplot as plt


def training(args, model, criterion, optimizer, device):
    _, dataloaders = create_dataloaders(args)
    epoch_num = args.num_epochs
    best_val_acc = 0
    val_loss_min = np.Inf

    logs_path = "./logs/v_" + str(args.version)
    if not os.path.exists(logs_path):
        os.makedirs(logs_path)

    output_file_name = "./output/v_" + str(args.version)
    if not os.path.exists(output_file_name):
        os.makedirs(output_file_name)

    tb_writer = SummaryWriter(logs_path)

    for epoch in range(1, epoch_num+1):

        model.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        total_loss_train, total_acc_train = [], []

        curr_iter = (epoch - 1) * len(dataloaders['train'])

        for i, data in tqdm(enumerate(dataloaders['train'])):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            N = images.size(0)
            # print('image shape:',images.shape, 'label shape',labels.shape)
            #images = Variable(images).to(device)
            #labels = Variable(labels).to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            prediction = outputs.max(1, keepdim=True)[1]
            train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)
            train_loss.update(loss.item())
            tb_writer.add_scalar("Train Loss", train_loss.avg, i)
            tb_writer.add_scalar("Train Accuracy", train_acc.avg, i)
            curr_iter += 1
            if (i + 1) % 100 == 0:
                print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                    epoch, i + 1, len(dataloaders['train']), train_loss.avg, train_acc.avg))
                total_loss_train.append(train_loss.avg)
                total_acc_train.append(train_acc.avg)
        loss_train = train_loss.avg
        acc_train = train_acc.avg
        tb_writer.add_scalar("Train Loss/epoch", loss_train, epoch)
        tb_writer.add_scalar("Train Accuracy/epoch", acc_train, epoch)

        # Validation
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()

        total_loss_val, total_acc_val = [], []

        with torch.no_grad():
            for i, data in enumerate(dataloaders['val']):
                images, labels = data
                N = images.size(0)
                images = Variable(images).to(device)
                labels = Variable(labels).to(device)

                outputs = model(images)
                prediction = outputs.max(1, keepdim=True)[1]

                val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)

                val_loss.update(criterion(outputs, labels).item())
                tb_writer.add_scalar("Val Loss", val_loss.avg, i)
                tb_writer.add_scalar("Val Accuracy", val_acc.avg, i)
        loss_val = val_loss.avg
        acc_val = val_acc.avg
        tb_writer.add_scalar("Val Loss/Epoch", loss_val, epoch)
        tb_writer.add_scalar("Val Accuracy/Epoch", acc_val, epoch)

        print('------------------------------------------------------------')
        print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
        print('------------------------------------------------------------')

        total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)

        if loss_val < val_loss_min:
            # Save model
            save_checkpoint(path=output_file_name, model=model, epoch=epoch, optimizer=optimizer)
            val_loss_min = loss_val
            print('*****************************************************')
            print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
            print('*****************************************************')


    tb_writer.close()

    #fig = plt.figure(num=2)
    #fig1 = fig.add_subplot(2, 1, 1)
    #fig2 = fig.add_subplot(2, 1, 2)
    #fig1.plot(total_loss_train, label='training loss')
    #fig1.plot(total_acc_train, label='training accuracy')
    #fig2.plot(total_loss_val, label='validation loss')
    #fig2.plot(total_acc_val, label='validation accuracy')
    #plt.legend()
    #plt.show()

'''
def train(train_loader, model, criterion, optimizer, epoch, device):
    model.train()

    train_loss = AverageMeter()
    train_acc = AverageMeter()

    total_loss_train, total_acc_train = [], []

    curr_iter = (epoch - 1) * len(train_loader)

    for i, data in enumerate(train_loader):
        images, labels = data
        N = images.size(0)
        # print('image shape:',images.shape, 'label shape',labels.shape)
        images = Variable(images).to(device)
        labels = Variable(labels).to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        prediction = outputs.max(1, keepdim=True)[1]
        train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)
        train_loss.update(loss.item())
        curr_iter += 1
        if (i + 1) % 100 == 0:
            print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))
            total_loss_train.append(train_loss.avg)
            total_acc_train.append(train_acc.avg)
    return train_loss.avg, train_acc.avg, total_loss_train, total_acc_train

'''
'''
def validate(val_loader, model, criterion, optimizer, epoch, device):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()

    with torch.no_grad():
        for _, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)

            val_loss.update(criterion(outputs, labels).item())

    print('------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))
    print('------------------------------------------------------------')
    return val_loss.avg, val_acc.avg
    
'''