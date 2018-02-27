from PIL import Image, ImageOps
from urllib.request import urlretrieve

import numpy as np
import os


vgg19_download_link = 'http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat'


def vgg19_download(file_name, expected_bytes=534904783):
    """ Download the pre-trained VGG-19 model if it's not already downloaded """

    if os.path.exists(file_name):
        print("[*] VGG-19 pre-trained model already exists")
        return

    print("[*] Downloading the VGG-19 pre-trained model...")

    file_name, _ = urlretrieve(vgg19_download_link, file_name)
    file_stat = os.stat(file_name)

    if file_stat.st_size == expected_bytes:
        print('[+] Successfully downloaded VGG-19 pre-trained model', file_name)
    else:
        raise Exception('[-] File ' + file_name + ' might be corrupted :(')


def image_resize(img_path, width, height, save=True):
    """ Resizing image with given size (h, w) """

    image = Image.open(img_path)  # open image

    image = ImageOps.fit(image, (width, height), Image.ANTIALIAS)  # resize

    if save:
        image_dirs = img_path.split('/')
        image_dirs[-1] = 'resized_' + image_dirs[-1]
        out_path = '/'.join(image_dirs)

        if not os.path.exists(out_path):
            image.save(out_path)
        else:
            print('[*] Image already exists')

    image = np.asarray(image, np.float32)

    return np.expand_dims(image, axis=0)


def setup_dir():
    """ Making required directory """

    os.mkdir('./content/')
    os.mkdir('./checkpoints/')
    os.mkdir('./style/')
    os.mkdir('./outputs/')
    os.mkdir('./graphs/')
