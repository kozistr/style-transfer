import tensorflow as tf
import numpy as np

import utils


class StyleTransfer:

    def __init__(self, content_image, style_image, width=333, heigth=250, channel=3):

        self.img_width = width
        self.img_height = heigth
        self.img_channel = channel

        self.content_img = utils.image_resize(content_image, self.img_width, self.img_height)
        self.style_img = utils.image_resize(style_image, self.img_width, self.img_height)
        self.initial_img = utils.generate_noise_image(self.content_img, self.img_width, self.img_height)

        self.content_w = 0.05
        self.style_w = 0.02
