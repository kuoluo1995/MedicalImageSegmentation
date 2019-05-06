from utils.example_tools.base import *


class TwoDExample(BaseExample):
    name = 'twoD'

    def _create_example_format(self):
        self.image_tool.transpose((1, 0, 2))
        self.label_tool.transpose((1, 0, 2))
        for idx in range(self.image_tool.shape[0]):
            image, image_shape = self.image_tool.get_data_and_shape(idx)
            label, label_shape = self.label_tool.get_data_and_shape(idx)
            feature = {
                'image/name': feature_to_bytes_list(self.image_tool.name),
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
                'image/name': tf.FixedLenFeature([], tf.string),
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

            return_feature = {'image': image, 'name': features['image/name'], 'index': features['extra/index']}
            return return_feature, label
