import pandas as pd
import numpy as np
import torch as th
import cv2 as cv
import shutil


def extract_split(partition):
    images_path = 'data/imgs/img_align_celeba'
    annotations_path = 'data/list_attr_celeba.txt'
    partitions_path = 'data/list_eval_partition.txt'

    split = 0 if partition == 'train' else (1 if partition == 'val' else 2)

    # indices = []
    # df = pd.read_csv(partitions_path, sep=' ', names=['image', 'split'], header=None)
    # for index, row in df.iterrows():
    #     if int(row['split']) == np.random.choice([0, 1, 2], p=[0.7, 0.15, 0.15]):
    #         img_name = row['image']
    #         inx = int(img_name.split('.')[0])
    #         indices.append(inx)

    annotations = pd.read_csv(annotations_path, sep='\\s+', header=1)
    annotations = annotations['Eyeglasses'].to_numpy().reshape(-1, 1)
    for i in range(1, len(annotations) + 1):
        img_path = f'{images_path}/{i:06}.jpg'
        y = annotations[i - 1]
        partition = np.random.choice(['train', 'val', 'test'], p=[0.7, 0.15, 0.15])
        if y == 1:
            shutil.copy(img_path, f'data/{partition}/eg')
        else:
            shutil.copy(img_path, f'data/{partition}/noeg')

        if i % 100 == 0:
            print(f'Processed {i + 1} images')


if __name__ == '__main__':
    extract_split('train')

