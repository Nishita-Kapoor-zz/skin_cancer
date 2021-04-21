# Import the necessary modules
import pandas as pd
from sklearn.model_selection import train_test_split
import os
from glob import glob
import seaborn as sns
colors = sns.color_palette()

# match image_id to path for all images in dataset
all_image_path = glob("/home/nishita/datasets/skin_mnist/*/*.jpg")
imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}

class_mapping = {0:'Actinic keratoses',
                 1:"Basal cell carcinoma",
                 2: 'Benign keratosis-like lesions',
                 3: 'Dermatofibroma',
                 4: 'Melanocytic nevi',
                 5: 'Vascular lesions',
                 6: 'dermatofibroma',
                 }


def get_data(data_path, imageid_path_dict):

    # dict of lesion code (dx) to name
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'
    }

    # Exploring the HAM10000_metadata

    df_original = pd.read_csv(os.path.join(data_path, 'HAM10000_metadata.csv'))  # read csv file
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)   # map image_id to its path
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)     # map dx to lesion type
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes  # assign class codes for lesion types

    df_original[['cell_type_idx', 'cell_type']].sort_values('cell_type_idx').drop_duplicates()

    # Creating a new dataframe df_undup that contains only the non-duplicate elements.
    df_undup = df_original.groupby('lesion_id').count()
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)

    # Create new column specifying duplicated or not
    def duplicated_or_not(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    df_original['duplicates'] = df_original['lesion_id']
    df_original['duplicates'] = df_original['duplicates'].apply(duplicated_or_not)

    # creating the un-duplicated dataframe
    df_undup = df_original[df_original['duplicates'] == 'unduplicated']

    # Create the validation dataframe
    labels = df_undup['cell_type_idx']  # labels
    _, df_temp = train_test_split(df_undup, test_size=0.3, random_state=42, stratify=labels)

    # Assign val or train status to lesion_ids
    def get_val_rows(x):
        val_list = list(df_temp['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    df_original['train_or_val'] = df_original['image_id']
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
    # filter out train rows
    df_train = df_original[df_original['train_or_val'] == 'train']

    # Split the validation dataframe further to create test and val
    df_val, df_test = train_test_split(df_temp, test_size=0.5, random_state=42)

    df_test = df_test.reset_index()
    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index()

    return df_train, df_val, df_test
