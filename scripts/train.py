from torch.utils.tensorboard import SummaryWriter
from data.dataloader import *


def training(args, model, criterion, optimizer, device):
    _, dataloaders = create_dataloaders(args)
    epoch_num = args.num_epochs
    best_val_acc = 0
    best_f1 = 0
    val_loss_min = np.Inf

    logs_path = "./output/logs/v_" + str(args.version)
    checkpoint_path = "./output/checkpoints/"
    create_folder(logs_path)
    create_folder(checkpoint_path)

    tb_writer = SummaryWriter(logs_path)

    for epoch in range(1, epoch_num+1):

        model.train()

        train_loss = AverageMeter()
        train_acc = AverageMeter()

        curr_iter = (epoch - 1) * len(dataloaders['train'])

        for i, data in tqdm(enumerate(dataloaders['train'])):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            N = images.size(0)

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
                    epoch, i + 1, len(dataloaders['train']), train_loss.avg, train_acc.avg))

        loss_train = train_loss.avg
        acc_train = train_acc.avg
        tb_writer.add_scalar("Train Loss/epoch", loss_train, epoch)
        tb_writer.add_scalar("Train Accuracy/epoch", acc_train, epoch)

        # Validation
        model.eval()
        val_loss = AverageMeter()
        val_acc = AverageMeter()
        val_f1 = AverageMeter()

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
                val_f1.update(avg_fscore(labels.cpu().numpy(), prediction.cpu().numpy()))

        loss_val = val_loss.avg
        acc_val = val_acc.avg
        f1_val = val_f1.avg
        tb_writer.add_scalar("Val Loss/Epoch", loss_val, epoch)
        tb_writer.add_scalar("Val Accuracy/Epoch", acc_val, epoch)
        tb_writer.add_scalar("Val F1 Score/Epoch", f1_val, epoch)

        print('------------------------------------------------------------')
        print('[epoch %d], [val loss %.5f], [val acc %.5f], [val f1 score %.5f]' % (epoch, loss_val, acc_val, f1_val))
        print('------------------------------------------------------------')

        if loss_val < val_loss_min:
            # if f1_val > best_f1:
            # Save model
            save_checkpoint(path=checkpoint_path + "checkpoint_v" + str(args.version) + ".pth", model=model, epoch=epoch, optimizer=optimizer)
            val_loss_min = loss_val
            # best_f1 = f1_val
            print('*****************************************************')
            print('best record: [epoch %d], [val loss %.5f], [val acc %.5f], [val f1 score %.5f]' % (epoch, loss_val, acc_val, f1_val))
            print('*****************************************************')

    tb_writer.close()
