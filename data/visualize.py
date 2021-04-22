import pandas as pd
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
colors = sns.color_palette()
from glob import glob
from utils import create_folder


def view_samples(args):
    # visualize some sample images
    all_image_path = glob(os.path.join(args.path, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}

    save_path = "./images/sample"
    create_folder(save_path)

    fig = plt.figure(figsize=(5, 5))
    columns, rows = 3, 2
    start, end = 0, len(imageid_path_dict)
    ax = []
    import random
    for i in range(columns*rows):
        k = random.randint(start, end)
        img = mpimg.imread(all_image_path[k])
        # create subplot and append to ax
        ax.append(fig.add_subplot(rows, columns, i+1))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
    plt.tight_layout()
    plt.title('Sample input images', fontdict={'size': 10})
    plt.savefig(save_path + 'input_image.png')

    # Checking the size and number of channels in the image
    arr = np.asarray(Image.open(all_image_path[10]))
    print(f"The shape of each image is {arr.shape}")


def data_dist(args):

    df_original = pd.read_csv(os.path.join(args.path, 'HAM10000_metadata.csv'))

    # Counts in each class
    plt.figure(figsize=(8, 5))
    ax = sns.countplot(x='dx', data=df_original, color=colors[0])
    plt.title('Class distribution', fontdict={'size': 15})
    plt.show()

    # Distribution of localization
    plt.figure(figsize=(18, 5))
    ax = sns.countplot(x='localization', data=df_original, color=colors[0])
    plt.title('Distribution of localization', fontdict={'size': 25})
    plt.show()

    # Distribution of age
    plt.figure(figsize=(8, 5))
    ax = sns.distplot(df_original['age'].dropna().values,
                      bins=10,
                      color=colors[0])
    plt.title('Distribution of Age', fontdict={'size': 20})
    plt.xlabel('age')
    plt.show()
