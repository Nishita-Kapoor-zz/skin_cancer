from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *
from data.data_analysis import *


def image_transform(norm_mean, norm_std):

    input_size = 224
    image_transforms = {
        # Train uses data augmentation
        'train': transforms.Compose([transforms.Resize((input_size, input_size)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomVerticalFlip(),
                                     transforms.RandomRotation(20),
                                     transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                     transforms.ToTensor(),
                                     transforms.Normalize(norm_mean, norm_std)]),

        'val': transforms.Compose([transforms.Resize((input_size, input_size)),
                                  transforms.ToTensor(),
                                  transforms.Normalize(norm_mean, norm_std)]),

        'test': transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(norm_mean, norm_std)])

    }

    return image_transforms


def create_dataloaders(args):

    df_train, df_val, df_test = get_data(args.path, imageid_path_dict)
    # norm_mean, norm_std = compute_img_mean_std(all_image_path)
    norm_mean = [0.7630401, 0.5456478, 0.57004625]   # change back later
    norm_std = [0.1409284, 0.1526128, 0.16997087]
    image_transforms = image_transform(norm_mean, norm_std)

    datasets = {
        'train': CustomDataset(df_train, transform=image_transforms['train']),
        'val': CustomDataset(df_val, transform=image_transforms['val']),
        'test': CustomDataset(df_test, transform=image_transforms['test'])
    }

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True, num_workers=2),
        'val': DataLoader(datasets['val'], batch_size=args.batch_size, shuffle=False, num_workers=2),
        'test': DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=False, num_workers=2)
    }

    return datasets, dataloaders
