from __future__ import absolute_import, division, print_function
import tensorflow as tf
import numpy as np
import os
import multiprocessing as mt
import random
from skimage import util
random.seed(0)

class MonodepthDataloader(object):
    def __init__(self, data_path, filenames_file, params, mode, num_output):
        self.params = params
        self.mode = mode
        self.num_output = num_output
        # 读取文件的路径存入列表中
        lists_and_labels = np.loadtxt(filenames_file, dtype=str)
        # 如果是训练模式则打乱顺序
        if self.mode == 'train':
            random.shuffle(lists_and_labels)
        image_path = lists_and_labels[:, :-2].tolist()
        pos_list = lists_and_labels[:, -2:].tolist()
        # 组合路径
        for i in range(len(image_path)):
            for j in range(len(image_path[i])):
                image_path[i][j] = os.path.join(data_path, image_path[i][j])
            for j in range(len(pos_list[i])):
                pos_list[i][j] = int(pos_list[i][j])

        total_num = len(lists_and_labels)

        dataset = tf.data.Dataset.from_tensor_slices((tf.constant(image_path), tf.constant(pos_list)))
        if self.mode == 'train':
            # 设置无限重复
            dataset = dataset.repeat()
            # 设置打乱顺序
            dataset = dataset.shuffle(buffer_size=total_num, seed=0)
        # 读取图片并做相应的数据增强和padding操作
        dataset = dataset.map(self._parse_image, num_parallel_calls=mt.cpu_count())
        if self.mode == 'train':
            # 设置batchsize
            dataset = dataset.batch(self.params.batch_size)
        elif self.mode == 'test':
            # 设置batchsize
            dataset = dataset.batch(1)
        # 计算每次迭代需要多少步
        # steps_per_epoch = np.ceil(len(labels) / TRAIN_BATCH_SIZE).astype(np.int32)
        self.dataset = dataset

    def _parse_image(self, files_path, pos):
        pos = tf.cast(pos, dtype=tf.float32)
        images = tf.map_fn(self.process_one_image, files_path, dtype=tf.float32)

        input_orig = images
        input_orig = tf.split(input_orig, num_or_size_splits=10, axis=0)
        input_orig = [tf.squeeze(image, axis=0) for image in input_orig]
        input_imgs_orig = tf.concat([input_orig[i] for i in range(9)], axis=2)

        images_noise = self.add_noise_random(images)
        images_noise = tf.split(images_noise, num_or_size_splits=10, axis=0)
        images_noise = [tf.squeeze(image, axis=0) for image in images_noise]



        # (h,w,27)
        input_imgs = tf.concat([images_noise[i] for i in range(9)], axis=2)
        label_img = images_noise[-1]
        onehot_pos = self.pos_to_onehot(pos)
        return input_imgs, label_img, pos, onehot_pos, input_imgs_orig

    def add_noise_random(self, image):
        return tf.py_func(self.add_noise_random_py, [image], tf.float32)

    def add_noise_random_py(self, image):
        # mode = random.randint(1, 11)
        # length = image.shape[0]

        # 高斯噪声
        # if mode == 1:
        #     sigma = random.uniform(0, 10)
        #     for i in range(length-1):
        #         image[i, :, :, :] = util.random_noise(image[i,:,:,:], 'gaussian', mean=0, var=(sigma/255)**2)
        # 椒盐噪声
        # if mode == 2:
        #     amount = random.uniform(0, 0.01)
        #     for i in range(length - 1):
        #         image[i, :, :, :] = util.random_noise(image[i, :, :, :], 's&p', amount=amount)

        return image



    def pos_to_onehot(self, pos):
        return tf.py_func(self.pos_to_onehot_py, [pos], tf.float32)

    def pos_to_onehot_py(self, pos):
        index = int(pos[1] * 7 + pos[0])
        onehot = np.zeros((self.params.height, 49, 1), dtype=np.float32)
        onehot[:, index, :] = 1
        return onehot

    def process_one_image(self, image_path):
            # 读取图片并解码
            image_string = tf.read_file(image_path)
            image_decoded = tf.image.decode_png(image_string, channels=3)
            # 由于无法从image_decoded推断shape，所以要先手动设定，否则resize_images会报错
            # image_decoded.set_shape([None, None, None])
            # 转换图像像素类型
            image_converted = tf.image.convert_image_dtype(image_decoded, tf.float32)
            # 裁切为352,512
            image_converted = image_converted[12:-12, 14:-15, :]
            # 调整大小
            image_resized = tf.image.resize_images(image_converted, [self.params.height, self.params.width],
                                                   tf.image.ResizeMethod.AREA)
            return image_resized


    def augment_image(self, image_list, num_output):
        # randomly shift gamma
        random_gamma = tf.random_uniform([], 0.8, 1.2)
        side_image_aug = [image_list[i] ** random_gamma for i in range(num_output)]

        # randomly shift brightness
        random_brightness = tf.random_uniform([], 0.5, 2.0)
        side_image_aug = [side_image_aug[i] * random_brightness for i in range(num_output)]

        # randomly shift color
        random_colors = tf.random_uniform([3], 0.8, 1.2)
        white_0 = tf.ones([tf.shape(image_list[0])[0], tf.shape(image_list[0])[1]])
        color_image_0 = tf.stack([white_0 * random_colors[i] for i in range(3)], axis=2)
        side_image_aug = [side_image_aug[i] * color_image_0 for i in range(num_output)]

        # saturate
        side_image_aug = [tf.clip_by_value(side_image_aug[i],  0, 1) for i in range(num_output)]

        return side_image_aug


    def adjustTone(self,image_list, num_output):
        img = [image_list[i] ** (1/1.5) for i in range(num_output)]
        img = [tf.image.rgb_to_hsv(img[i]) for i in range(num_output)]
        img = [img[i] * [1, 1.5, 1] for i in range(num_output)]
        img = [tf.image.hsv_to_rgb(img[i]) for i in range(num_output)]
        img = [tf.clip_by_value(img[i], 0, 1) for i in range(num_output)]
        return img
