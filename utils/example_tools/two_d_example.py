from utils.example_tools.base import *


class TwoDExample(BaseExample):
    name = 'twoD'
    extra_feature = {'index': {CustomKeys.VALUE: 0, CustomKeys.TYPE: tf.int64}}

    def _create_example_format(self):
        self.dataset_class.deal_image(self.image_reader)
        self.dataset_class.deal_image(self.label_reader)
        for idx in range(self.image_reader.shape[0]):
            image, image_shape = self.image_reader.get_data_and_shape(idx)
            label, label_shape = self.label_reader.get_data_and_shape(idx)
            feature = {
                'image/image_path': feature_to_bytes_list(self.image_reader.image_path),
                'image/shape': feature_to_int64_list(image_shape),
                'image/encoded': feature_to_bytes_list(image),
                'segmentation/shape': feature_to_int64_list(label_shape),
                'segmentation/encoded': feature_to_bytes_list(label),
                'extra/index': feature_to_int64_list(idx)
            }
            yield tf.train.Example(features=tf.train.Features(feature=feature))

    def read_example(self, example_proto, mode, **params):  # todo improve 这里参数似乎可以改变下。
        tf.logging.info('................>>>>>>>>>>>>>>>> reading {} example'.format(self.name))
        with tf.variable_scope('ParseExample'):
            features = {
                'image/image_path': tf.FixedLenFeature([], tf.string),
                'image/shape': tf.FixedLenFeature([3], tf.int64),
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'segmentation/shape': tf.FixedLenFeature([2], tf.int64),
                'segmentation/encoded': tf.FixedLenFeature([], tf.string),
                'extra/index': tf.FixedLenFeature([], tf.int64)
            }
            features = tf.parse_single_example(example_proto, features=features)
            with tf.variable_scope('GetImage'):
                image = tf.decode_raw(features['image/encoded'], tf.int16)
                image = tf.reshape(image, features['image/shape'])
                image = tf.to_float(image)
            with tf.variable_scope('GetLabel'):
                label = tf.decode_raw(features['segmentation/encoded'], tf.uint8)
                label = tf.reshape(label, features['segmentation/shape'])
                label = tf.to_int32(label)
            feature = {CustomKeys.IMAGE: image, CustomKeys.IMAGE_PATH: features['image/image_path'],
                       'index': features['extra/index']}
            for other, value in params['extra_feature'].items():
                if other not in feature:
                    feature[other] = tf.constant(value[CustomKeys.VALUE], dtype=value[CustomKeys.TYPE])
            return feature, label,
