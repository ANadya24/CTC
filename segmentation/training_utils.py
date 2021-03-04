import numpy as np
from tensorflow.python.keras.layers import *

# generates patches of size (patch_size, patch_size) from img and binary (!) label, where patches are taken from 'border_offset' pixels from border and with 'stride'
# by default, border_offset is 0 and stride = patch_size.
# If adaptive_stride is True, whole image will be cut to patches excluding border offset, using max possible stride for this size
def patch_generator(img, label, patch_size, label_percentage_threshold, stride = None, border_offset = 0, adaptive_stride = False):
    width = img.shape[1]
    height = img.shape[0]
    stride_x = patch_size
    stride_y = patch_size
    if stride is not None:
        stride_x = stride
        stride_y = stride

    dx = 0
    dy = 0
    if width % patch_size != 0: dx = 1
    if height % patch_size != 0: dy = 1
    n_patches_x = (width - 2 * border_offset) // patch_size + dx
    n_patches_y = (height - 2 * border_offset) // patch_size + dy

    if adaptive_stride:
        if n_patches_x == 1:
            stride_x = patch_size
        else:
            stride_x = (width - 2*border_offset - patch_size) // (n_patches_x - 1)
        if n_patches_y == 1:
            stride_y = patch_size
        else:
            stride_y = (height - 2*border_offset - patch_size) // (n_patches_y - 1)
    # off + x * stride + patch_size = size - off
    # stride = (size - 2*off - patch_size) // x where x is number of patches. Namely, size // patch_size + 1 or +0 if size = n * patch_size
    p_x = 0
    p_y = 0
    if (width - 2*border_offset) % n_patches_x == 0: p_x = 1
    if (height - 2*border_offset) % n_patches_y == 0: p_y = 1
    top_left_x = range(border_offset, width - border_offset - patch_size + p_x, stride_x)
    top_left_y = range(border_offset, height - border_offset - patch_size + p_y, stride_y)
    for x in top_left_x:
        for y in top_left_y:
            label_patch = label[y:y+patch_size, x:x+patch_size]
            avrg = np.average(label_patch)
            if avrg > label_percentage_threshold:
                yield img[y:y+patch_size, x:x+patch_size], label_patch

# normalize to [0,1]
def normalize(img):
    min = img.min()
    max = img.max()
    return (img - min) / (max - min)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


from tensorflow.python.keras.preprocessing.image import ImageDataGenerator


# training generator that modifies both x and y
def train_generator(x_train, y_train, batch_size=32):
    data_gen_args = dict(
        shear_range=10,
        rotation_range=90,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='reflect')

    datagen_data = ImageDataGenerator(**data_gen_args)
    datagen_label = ImageDataGenerator(**data_gen_args)

    seed = 1
    # image_datagen.fit(images, augment=True, seed=seed)
    # mask_datagen.fit(masks, augment=True, seed=seed)

    data_generator = datagen_data.flow(x_train, batch_size=batch_size, seed=seed)
    label_generator = datagen_label.flow(y_train, batch_size=batch_size, seed=seed)
    train_generator = zip(data_generator, label_generator)
    for x, y in train_generator:
        yield x, y
