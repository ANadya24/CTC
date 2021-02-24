import torch
from torchvision.transforms import ToTensor
from torch.utils import data
from skimage.transform import resize
import skimage.io as io
import numpy as np
import pickle
from scipy import ndimage
import cv2
import os
import albumentations as A


def pad(img, padwidth):
    h0, h1, w0, w1 = padwidth[:4]
    if len(padwidth) < 5:
        img = np.pad(img, ((h0, h1), (w0, w1)))
    else:
        img = np.pad(img, ((h0, h1), (w0, w1), (0, 0)))

    return img

def normalize(im):
    im = im - im.min()
    im = im / (im.max() + 1e-10)
    return im


class Dataset(data.Dataset):
    def __init__(self, path, im_size=(1, 100, 100), smooth=False, train=True, shuffle=True, use_crop=False):
        """Initialization"""
        self.path = path
        self.folders = os.listdir(path)

        self.length = sum([len(os.listdir(path+f)) for f in self.folders])
        print('Dataset length is ', self.length)
        self.seq = []

        for f in self.folders:
            for i, im in enumerate(os.listdir(path+f)):
                self.seq.append((path+f+'/'+im, i))

        # self.seq = self.seq[:5]
        # self.length = len(self.seq)
        self.use_crop = use_crop
        self.im_size = im_size[1:]
        self.smooth = smooth
        self.train = train
        self.shuffle = shuffle
        if self.shuffle:
            np.random.shuffle(self.seq)

        if self.train:
             self.aug_pipe = A.Compose([A.HorizontalFlip(p=0.3), A.VerticalFlip(p=0.3),
                                   A.ShiftScaleRotate(shift_limit=0.0225, scale_limit=0.1, rotate_limit=15, p=0.2)], 
                                   additional_targets={'image2': 'image'})

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        im_path, it = self.seq[index]
        number = int(im_path.split('crop')[-1].split('.')[0])
        rank = 3 
        if self.train:
            number2 = np.random.randint(number-3, number+3)
            if not os.path.exists(im_path.split('crop')[0] + f'crop{number2}.tif'):
                if os.path.exists(im_path.split('crop')[0] + f'crop{number+1}.tif'):
                    number2 = number + 1
                elif os.path.exists(im_path.split('crop')[0] + f'crop{number-1}.tif'):
                    number2 = number - 1
                else:
                    number2 = number
        else:
            if os.path.exists(im_path.split('crop')[0] + f'crop{number+1}.tif'):
                number2 = number + 1
            elif os.path.exists(im_path.split('crop')[0] + f'crop{number-1}.tif'):
                number2 = number - 1
            else:
                number2 = number
#         print(number, number2)
        
#        if number2 < number:
#            tmp = number
#            number = number2
#            number2 = tmp

#        to_tensor = ToTensor()
#        print(im_path)
        fixed_image = cv2.imread(im_path, -1)
        moving_image = cv2.imread(im_path.split('crop')[0] + f'crop{number2}.tif', -1)
        if im_path.find('jpn') > 0:
            fixed_image = fixed_image.max() - fixed_image
            moving_image = moving_image.max() - moving_image
        h, w = fixed_image.shape[:2]

        fixed_image = normalize(fixed_image)
        moving_image = normalize(moving_image)
        h, w = fixed_image.shape[:2]
        if h != w:
            if h < w:
                fixed_image = pad(fixed_image, ((w-h)//2, w-(w-h)//2, 0, 0))
            else:
                fixed_image = pad(fixed_image, (0, 0, (h-w)//2, h-(h-w)//2))
        h, w = moving_image.shape[:2]
        if h != w:
            if h < w:
                moving_image = pad(moving_image, ((w-h)//2, w-(w-h)//2, 0, 0))
            else:
                moving_image = pad(moving_image, (0, 0, (h-w)//2, h-(h-w)//2))
#        print(fixed_image.shape, fixed_image.max())
        # Load data and get label
        # fixed_image = io.imread(ID + '_1.jpg', as_gray=True)
#        fixed_image = self.data[seqID]['imseq'][it].astype('uint8')
        if self.use_crop:
            x0 = np.random.randint(0, w - self.im_size[1])
            y0 = np.random.randint(0, h - self.im_size[0])
            fixed_image = fixed_image[y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
            moving_image = moving_image[y0: y0 + self.im_size[0], x0:x0 + self.im_size[1]]
        else:
            c = min(h, w)
            fixed_image = fixed_image[:c, :c]
            moving_image = moving_image[:c, :c]
            fixed_image = cv2.resize(fixed_image, tuple(self.im_size))
            moving_image = cv2.resize(moving_image, tuple(self.im_size))

        if self.train:
            arr = self.aug_pipe(image=fixed_image, image2=moving_image)
            fixed_image = arr['image']
            moving_image = arr['image2']
        fixed_image = torch.Tensor(fixed_image[None]).float()
        moving_image = torch.Tensor(moving_image[None]).float()
        return fixed_image, moving_image


if __name__ == '__main__':
    from glob import glob
    from matplotlib import pyplot as plt

    path = '/data/sim/CTC/data/Hela/train/'
    dataset = Dataset(path, (1, 128, 128),
                      smooth=True, train=True, shuffle=True)
    fixed, moving = dataset[0]
    print(fixed.shape, fixed.max())
    fixed = np.uint8(fixed.numpy().transpose((1,2,0))*255)
    moving = np.uint8(moving.numpy().transpose((1,2,0))*255)
    tmp = np.concatenate([fixed, moving], axis=1)
    print(tmp.shape)
    cv2.imwrite('test1.jpg', tmp)
