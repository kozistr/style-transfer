# Some codes in this file was borrowed from https://github.com/chiphuyen/stanford-tensorflow-tutorials


import tensorflow as tf
import argparse
import time
import os

import utils
import vgg19


def kwargs():
    description = "Tensorflow implementation of Image Style Transfer (Neural Style)"

    parser = argparse.ArgumentParser(description=description)

    parser.add_argument('--content', type=str, default='contents/deadpool.jpg', required=True,
                        help='file path of content image')
    parser.add_argument('--style', type=str, default='styles/guernica.jpg', required=True,
                        help='file path of style image')

    parser.add_argument('--content_w', type=int, default=0.05,
                        help='weight of content loss')
    parser.add_argument('--style_w', type=int, default=0.02,
                        help='weight of style loss')

    parser.add_argument('--image_width', type=int, default=333,
                        help='width size of the images')
    parser.add_argument('--image_height', type=int, default=250,
                        help='height size of the images')

    parser.add_argument('--train_steps', type=int, default=500,
                        help='training epoch')

    return parser.parse_args()


class StyleTransfer:

    def __init__(self, content_image, style_image, width, height, channel,
                 content_w, style_w,
                 training_steps, logging_steps=1):

        self.img_width = width
        self.img_height = height
        self.img_channel = channel
        self.input_image = None

        self.content_img = utils.image_resize(content_image, self.img_width, self.img_height, save=False)
        self.style_img = utils.image_resize(style_image, self.img_width, self.img_height, save=False)
        self.initial_img = utils.generate_noise_image(self.content_img, self.img_width, self.img_height)

        self.content_layer = ['conv4_2']
        self.style_layers = ['conv1_1', 'conv2_1', 'conv3_1', 'conv4_1', 'conv5_1']

        self.content_w = content_w
        self.style_w = style_w

        self.content_layer_w = [1.0]
        self.style_layer_w = [0.5, 1.0, 1.5, 3.0, 4.0]  # [0.2] * 5

        self.vgg19 = None  # VGG19 model

        self.content_loss = 0.
        self.style_loss = 0.
        self.total_loss = 0.

        # Hyper-Parameters
        self.train_steps = training_steps
        self.logging_steps = logging_steps
        self.g_step = tf.Variable(0, trainable=False, name='global_steps')
        self.opt = None
        self.summary = None
        self.lr = 1.

        self.build()

    def create_input_image(self):
        # create input image 'tensor' variable
        self.input_image = tf.get_variable('input_image',
                                           shape=(1, self.img_height, self.img_width, self.img_channel),
                                           initializer=tf.zeros_initializer(),
                                           dtype=tf.float32)

    def build_vgg19(self):
        self.vgg19 = vgg19.VGG19(self.input_image)  # load VGG19 model

        self.content_img -= self.vgg19.mean_pixels  # normalize
        self.style_img -= self.vgg19.mean_pixels    # normalize

    def _gram_matrix(self, f, n, m):
        f = tf.reshape(f, (m, n))
        return tf.matmul(tf.transpose(f), f)

    def _content_loss(self, p, f):
        self.content_loss = tf.reduce_sum(tf.square(f - p)) / (4. * p.size)

    def _single_style_loss(self, a, g):
        n = a.shape[3]
        m = a.shape[1] * a.shape[2]

        a = self._gram_matrix(a, n, m)
        g = self._gram_matrix(g, n, m)

        return tf.reduce_sum(tf.square(g - a)) / (2 * n * m) ** 2

    def _style_loss(self, img):
        n_layers = len(self.style_layers)
        e = [self._single_style_loss(img[i], self.vgg19.vgg19_net[self.style_layers[i]]) for i in range(n_layers)]

        self.style_loss = sum([self.style_layer_w[i] * e[i] for i in range(n_layers)])

    def losses(self):
        with tf.Session() as s:
            s.run(self.input_image.assign(self.content_img))

            gen_img_content = self.vgg19.vgg19_net[self.content_layer[0]]
            content_img_content = s.run(gen_img_content)
            self._content_loss(content_img_content, gen_img_content)

            s.run(self.input_image.assign(self.style_img))
            style_layers = s.run([self.vgg19.vgg19_net[layer] for layer in self.style_layers])
            self._style_loss(style_layers)

        self.total_loss = self.content_w * self.content_loss + self.style_w * self.style_loss

    def build(self):
        """ Building Style-Transfer Model """

        self.create_input_image()
        self.build_vgg19()
        self.losses()

    def train(self):
        # Optimizer
        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=self.total_loss,
                                                                          global_step=self.g_step)

        # Summaries
        tf.summary.scalar('content_loss', self.content_loss)
        tf.summary.scalar('style_loss', self.style_loss)
        tf.summary.scalar('total_loss', self.total_loss)

        self.summary = tf.summary.merge_all()

        # GPU configure
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        with tf.Session(config=config) as s:
            # Saver & Writer
            saver = tf.train.Saver(max_to_keep=1)

            s.run(tf.global_variables_initializer())
            writer = tf.summary.FileWriter('graphs', s.graph)

            s.run(self.input_image.assign(self.initial_img))

            ckpt = tf.train.get_checkpoint_state(os.path.dirname('./checkpoints/checkpoint'))
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(s, ckpt.model_checkpoint_path)

            initial_step = self.g_step.eval()

            start_time = time.time()
            for idx in range(initial_step, initial_step + self.train_steps + 1):
                if 5 <= idx < 50:
                    self.logging_steps = 15
                elif idx >= 50:
                    self.logging_steps = 50

                s.run(self.opt)  # Train

                if (idx + 1) % self.logging_steps == 0:
                    gen_image, total_loss, summary = s.run([self.input_image, self.total_loss, self.summary])

                    gen_image += self.vgg19.mean_pixels
                    writer.add_summary(summary, global_step=idx)

                    print('[*] Step : {} loss : {:5.2f}'.format(idx + 1, total_loss))
                    print('    Took : {} seconds'.format(time.time() - start_time))

                    start_time = time.time()

                    filename = './outputs/' + content.split('/')[-1] + '_' + style.split('/')[-1] + '_%d.png' % idx
                    utils.image_save(gen_image, filename)

                    if (idx + 1) % 20 == 0:
                        saver.save(s, './checkpoints/style_transfer', idx)


if __name__ == '__main__':
    args = kwargs()
    if args is None:
        exit(0)

    utils.setup_dir()

    st = StyleTransfer(content_image=args.content, style_image=args.style,
                       width=args.image_width, height=args.image_height, channel=3,
                       content_w=args.content_w, style_w=args.style_w,
                       training_steps=args.train_steps)

    st.train()  # train style-transfer
