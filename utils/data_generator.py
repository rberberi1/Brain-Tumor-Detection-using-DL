import os
import numpy as np

def load_img(img_dir, img_list):
    """
    Loads a list of .npy image files from a directory into a NumPy array.
    """
    images = []
    for file in img_list:
        if file.endswith('.npy'):
            image = np.load(os.path.join(img_dir, file))
            images.append(image)
    return np.array(images)


def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    """
    Custom data generator to yield batches of images and masks from .npy files.
    Designed for 3D input data in BraTS (128x128x128x3).
    """
    L = len(img_list)

    while True:
        batch_start = 0
        batch_end = batch_size

        while batch_start < L:
            limit = min(batch_end, L)

            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])

            yield (X, Y)

            batch_start += batch_size
            batch_end += batch_size
