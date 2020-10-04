import numpy as np
import rasterio
import cv2
import imgaug as ia
import tensorflow as tf

from PIL import Image
from glob import glob
from imgaug import augmenters as iaa

from tensorflow.keras.preprocessing.image import ImageDataGenerator




class UnetGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'

    def __init__(self, data_path,
                 to_fit=True, batch_size=1, dim=(256, 256),
                 n_channels=6, shuffle=True, augment=False):
        'Initialization'
        ia.seed(2)

        self.data_path = data_path
        self.tif_list = []
        self.mask_list = []
        for filename in glob(f'{data_path}*.tif*'):
            self.tif_list.append(filename)

        self.to_fit = to_fit
        self.augment = augment
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.n = 0
        self.max = self.__len__()
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.tif_list) / self.batch_size))

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.tif_list))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index *
                               self.batch_size: (index + 1) * self.batch_size]

        # Find list of IDs
        tif_list_temp = [self.tif_list[k] for k in indexes]

        # Generate data
        X = self._generate_X(tif_list_temp)

        #  preprocess and augment data

        if self.to_fit:
            y = self._generate_y(tif_list_temp)

            if self.augment:
                seq = make_augmentations()
                images_aug = list()
                labels_aug = list()
                for i in range(len(X)):
                    image, label = seq(
                        image=X[i].astype('float32'),
                        segmentation_maps=np.expand_dims(
                            y[i], 0).astype('uint8')
                    )
                    images_aug.append(image)
                    labels_aug.append(label[0, :, :, :])

                return np.array(images_aug), np.array(labels_aug)
            else:
                # import ipdb; ipdb.set_trace()
                return X, y[:,:,:,0]

        else:
            return X

    def _generate_X(self, tif_list_temp):
        'Generates data containing batch_size images'
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))

        # Generate data
        for i, ID in enumerate(tif_list_temp):
            # Store sample
            X[i, ] = _load_tif_image(ID, self.dim)
        return X

    def _generate_y(self, tif_list_temp):
        'Generates data containing batch_size masks'
        y = np.empty((self.batch_size, *self.dim, 1), dtype='float32')

        # Generate data
        for i, ID in enumerate(tif_list_temp):
            # replace tif/tiff with the _bitmap.png or .bmp, depending on the dataset
            # to look for corresponding masks for the imput images
            y[i, ] = _load_grayscale_image(ID.replace(
                '.tif', '.bmp'), self.dim
            )
        return y

    def __next__(self):
        if self.n >= self.max:
            self.n = 0
        result = self.__getitem__(self.n)
        self.n += 1
        return result

def _load_grayscale_image(image_path, dim):
    'Load grayscale image'
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255

    return np.expand_dims(cv2.resize(img, dim), -1)


def _load_tif_image(image_path, dim):
    'load tif image'

    with rasterio.open(image_path, 'r') as data:
        return cv2.resize(
                    np.moveaxis(data.read(), 0, -1), dim
                )


def sometimes(aug):
    return iaa.Sometimes(0.5, aug)

def make_augmentations():

    return iaa.Sequential([
        sometimes(iaa.CoarseDropout(0.1, size_percent=0.2)),
        sometimes(
            iaa.Affine(
                scale={"x": (0.9, 1.1), "y": (0.9, 1.1)},
                # scale images to 80-120% of their size,
                # individually per axis
                translate_percent={
                    "x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                # translate by -20 to +20 percent (per axis)
                # rotate by -45 to +45 degrees
                rotate=(-10, 10),
                shear=(-5, 5),  # shear by -16 to +16 degrees
            ),
        ),
        sometimes(iaa.ElasticTransformation(alpha=10, sigma=1))
    ],
        random_order=True
    )