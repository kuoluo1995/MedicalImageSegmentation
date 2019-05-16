import numpy as np
from pathlib import Path
from tensorflow import Dimension, TensorShape

from model_component.mode.base import *
from utils import reader_tools


class TrainOrganMode(BaseMode):
    name = 'train_organ'

    def set_config(self, **params):
        tf.logging.info('..... setting {} mode config'.format(self.name))
        self.dataset_path = params['dataset_path']

    def get_dataset(self, **params):
        def generator():
            while True:
                for fold in Path(self.dataset_path).iterdir():
                    image = reader_tools.nii_reader(str(fold) + '/image.nii').astype(np.float32, copy=False)
                    label = reader_tools.nii_reader(str(fold) + '/label.nii').astype(np.float32, copy=False)
                    if image.shape != label.shape:
                        raise RuntimeError("Shape mismatched between image and label: {} vs {},image_path:{} ".format(
                            image.shape, label.shape, str(fold) + '/image.nii'))
                    image, label = image_process_operations.random_volume_zoom_in(image[..., np.newaxis], label, 1.5,
                                                                                  params['seed'])
                    for i in range(image.shape[0]):
                        yield {'image': image[i] / np.max(image),
                               'name': str(fold) + '/image.nii', 'index': i}, (label[i] + 0.5).astype(np.uint8,
                                                                                                      copy=False)

        with tf.variable_scope('TrainOrganDataset'):
            tf.logging.info('..... get {} dataset'.format(self.name))
            dataset = tf.data.Dataset().from_generator(generator, output_types=(
                {'image': tf.float32, 'name': tf.string, 'index': tf.int64}, tf.int32), output_shapes=(
                {'image': TensorShape([Dimension(None), Dimension(256), Dimension(256), Dimension(None)]),
                 'name': TensorShape([Dimension(None)]), 'index': TensorShape([Dimension(None)])},
                TensorShape([Dimension(None), Dimension(256), Dimension(256)])))
            return dataset

    def init_iterator(self, dataset):
        self.iterator = dataset.make_one_shot_iterator()
        return None

    def adjust_window_size(self, image, min_window, max_window):
        min_window_tensor = tf.random_uniform([], -50, 50)
        max_window_tensor = tf.random_uniform([], -50, 50)
        new_min_window = tf.add(float(min_window), min_window_tensor, name='min_window')
        new_max_window = tf.add(float(max_window), max_window_tensor, name='max_window')
        image = image_process_operations.adjust_window_size(image, new_min_window, new_max_window)
        return image
