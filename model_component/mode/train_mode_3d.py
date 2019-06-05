import numpy as np
from pathlib import Path
from tensorflow import Dimension, TensorShape

from model_component.mode.base import *
from utils import reader_tools


class TrainMode3D(BaseMode):
    name = 'train_organ'

    def set_config(self, **params):
        tf.logging.info('..... setting {} mode config'.format(self.name))
        self.dataset_path = params['dataset_path']

    def get_dataset(self, **params):
        def data_preprocess(example_proto):
            with tf.variable_scope('InputPipeline/'):
                # 将example转化成数据
                feature, label3d, = self.example.read_example(example_proto, self, **params)
                with tf.variable_scope('Augmentation'):
                    # todo test 数据增广对函数的影响和其他方法
                    if params['image_augmentation'] is not None:
                        for function_name in params['image_augmentation']:
                            kwargs = {} if params['image_augmentation'][function_name] == 'None' else \
                                params['image_augmentation'][function_name]
                            feature[CustomKeys.IMAGE], new_label = image_process_operations.process_image(
                                function_name=function_name, image=feature[CustomKeys.IMAGE], label=label3d, **kwargs,
                                **params)
                            if new_label is not None:
                                label3d = new_label

                with tf.variable_scope('UnitedImageSize'):
                    image3d = tf.image.resize_bilinear(tf.expand_dims(feature[CustomKeys.IMAGE], axis=-1),
                                                       [params['image_height'], params['image_width']])
                    feature[CustomKeys.IMAGE] = self.adjust_window_size(image3d, params['min_window_level'],
                                                                        params['max_window_level'])
                with tf.variable_scope('UnitedLabelSize'):
                    label3d = tf.image.resize_nearest_neighbor(tf.expand_dims(label3d, axis=-1),
                                                               [params['image_height'], params['image_width']])
                    label3d = tf.squeeze(label3d, axis=-1)
                    label3d = tf.clip_by_value(label3d, 0, 1)  # todo improve 之后多类别同时识别考虑修改
                return feature, label3d

        def flat_map(feature, label3d):
            def _map(each_image, each_label):
                feature[CustomKeys.IMAGE] = each_image
                return feature, each_label

            with tf.name_scope('InputPipeline/'):
                shape = tf.concat((tf.shape(feature[CustomKeys.IMAGE])[:-1], [self.dataset_dict['image_channel'] // 2]),
                                  axis=0)
                padding_image = tf.zeros(shape, dtype=feature[CustomKeys.IMAGE].dtype)
                image_multi_channels = tf.concat((padding_image, feature[CustomKeys.IMAGE], padding_image), axis=-1)
                images = Dataset.from_tensor_slices(image_multi_channels)
                labels = Dataset.from_tensor_slices(label3d)
                return Dataset.zip((images, labels)).map(_map)

        def data_preprocess_2d(feature, label):
            image = tf.image.resize_bilinear(tf.expand_dims(feature[CustomKeys.IMAGE], axis=0),
                                             [params['image_height'], params['image_width']])
            feature[CustomKeys.IMAGE] = tf.squeeze(image, axis=0)
            return feature, label

        with tf.variable_scope('Train3DDataset'):
            tf.logging.info('..... get {} dataset'.format(self.name))
            # todo improve 找一种一次性拿出来的方法
            dataset = tf.data.TFRecordDataset(self.dataset_dict['dataset'][0])
            for file_name in self.dataset_dict['dataset'][1:]:
                dataset = dataset.concatenate(tf.data.TFRecordDataset(file_name))
            dataset = (dataset.prefetch(buffer_size=1).repeat(count=None)
                       .map(data_preprocess).flat_map(flat_map)
                       .apply(tf.data.experimental.map_and_batch(data_preprocess_2d, params['batch_size'],
                                                                 num_parallel_batches=params['num_parallel_batches']))
                       .prefetch(buffer_size=contrib_data.AUTOTUNE))
            return dataset

    def adjust_window_size(self, image, min_window, max_window):
        min_window_tensor = tf.random_uniform([], -50, 50)
        max_window_tensor = tf.random_uniform([], -50, 50)
        new_min_window = tf.add(float(min_window), min_window_tensor, name='min_window')
        new_max_window = tf.add(float(max_window), max_window_tensor, name='max_window')
        image = image_process_operations.adjust_window_size(image, new_min_window, new_max_window)
        return image

    def get_extra_feature(self, **params):
        self.dataset_dict = yaml_tools.read(params['base_path'] / self.dataset_path)
        self.example = example_tools.create_example(self.dataset_dict['example'])
        return self.example.extra_feature
