import pandas
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from torch.autograd import Variable

from data.dataloader import *

'''
def predict(**cfg):

    # Create a test loader
    loader = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])
    ])

    # Load the image as a Pytorch Tensor
    image = loader(Image.open(cfg["predict"]["image_path"]).convert("RGB")).float().unsqueeze(0)

    # Load Model
    model = load_checkpoint(**cfg)

    # Check GPU availability
    train_gpu, _ = check_gpu()
    if train_gpu:
        image = image.cuda()

    # Get model prediction
    output = model(image)
    _, pred = torch.max(output, 1)

    # Check prediction - Normal or Pneumonia
    print("Prediction: " + str(model.idx_to_class[pred]))
'''


def evaluate(args, device, model):
    _, dataloaders = create_dataloaders(args)
    checkpoint_path = "./output/checkpoints/checkpoint_v" + str(args.version) + ".pth"
    checkpoint = load_checkpoint(path=checkpoint_path, model=model)
    save_path = "./output/metrics/"
    create_folder(save_path)

    fig_save = save_path + "confusion_matrix_v" + str(args.version) + ".png"
    csv_save = save_path + "report_v" + str(args.version) + ".csv"

    model = checkpoint["model"]
    model = model.to(device)

    model.eval()
    y_label = []
    y_predict = []
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloaders['test'])):
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

    plot_confusion_matrix(confusion_mtx, fig_path=fig_save, classes=plot_labels)

    report = classification_report(y_label, y_predict, target_names=plot_labels, output_dict=True)
    report_df = pandas.DataFrame(report).transpose()
    report_df.to_csv(csv_save, index=False)
