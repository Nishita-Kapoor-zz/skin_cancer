from torchvision import transforms
from torch.utils.data import DataLoader
from utils import *
import os
from glob import glob
from data.data_analysis import *


base_dir = args.path
all_image_path = glob(os.path.join(base_dir, '*', '*.jpg'))
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}

input_size = 224

df_train, df_val, df_test = get_data(base_dir, imageid_path_dict)

normMean, normStd = compute_img_mean_std(all_image_path)


train_transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.RandomRotation(20),
                                      transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                      transforms.ToTensor(),
                                      transforms.Normalize(normMean, normStd)])

# define the transformation of the val images.
val_transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(normMean, normStd)])

# define the transformation of the test images.
test_transform = transforms.Compose([transforms.Resize((input_size, input_size)),
                                    transforms.ToTensor(),
                                    transforms.Normalize(normMean, normStd)])

# Define the training set using the table train_df and using our defined transitions (train_transform)
training_set = CustomDataset(df_train, transform=train_transform)
train_loader = DataLoader(training_set, batch_size=8, shuffle=True, num_workers=4)

# Same for the validation set:
validation_set = CustomDataset(df_val, transform=val_transform)
val_loader = DataLoader(validation_set, batch_size=8, shuffle=False, num_workers=4)

# Same for the validation set:
test_set = CustomDataset(df_test, transform=test_transform)
test_loader = DataLoader(test_set, batch_size=8, shuffle=False, num_workers=4)
