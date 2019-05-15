import tensorflow as tf

from utils.example_tools import base


class ThreeDExample(base.BaseExample):
    name = 'threeD'

    def _create_example_format(self):
        self.dataset_class.deal_image(self.image_reader)
        self.dataset_class.deal_image(self.label_reader)
        image, image_shape = self.image_reader.get_data_and_shape()
        label, label_shape = self.label_reader.get_data_and_shape()
        feature = {
            'image/image_path': base.feature_to_bytes_list(self.image_reader.image_path),
            'image/shape': base.feature_to_int64_list(image_shape),
            'image/encoded': base.feature_to_bytes_list(image),
            'segmentation/shape': base.feature_to_int64_list(label_shape),
            'segmentation/encoded': base.feature_to_bytes_list(label),
        }
        yield tf.train.Example(features=tf.train.Features(feature=feature))

    def read_example(self, example_proto, mode, **params):
        def padding(image, shape):
            padding_length = params['batch_size'] - shape[0] % params['batch_size']
            padding_shape = tf.concat(([padding_length], shape[1:]), axis=0)
            padding_image = tf.zeros(padding_shape, dtype=image.dtype)
            new_image = tf.concat((image, padding_image), axis=0)
            return new_image

        tf.logging.info('................>>>>>>>>>>>>>>>> reading {} example'.format(self.name))
        with tf.variable_scope('ParseExample'):
            features = {
                'image/image_path': tf.FixedLenFeature([], tf.string),
                'image/shape': tf.FixedLenFeature([3], tf.int64),
                'image/encoded': tf.FixedLenFeature([], tf.string),
                'segmentation/shape': tf.FixedLenFeature([3], tf.int64),
                'segmentation/encoded': tf.FixedLenFeature([], tf.string),
            }
            # with tf.control_dependencies([tf.print(features['image/name'], features['image/shape'])]):
            features = tf.parse_single_example(example_proto, features=features)
            with tf.variable_scope('GetImage'):
                image = tf.decode_raw(features['image/encoded'], tf.int16)
                image = tf.reshape(image, features['image/shape'])
                image = tf.cast(image, tf.float32)
                image = padding(image, features['image/shape'])
            with tf.variable_scope('GetLabel'):
                label = tf.decode_raw(features['segmentation/encoded'], tf.uint8)
                label = tf.reshape(label, features['segmentation/shape'])
                label = tf.cast(label, tf.int32)
                label = padding(label, features['segmentation/shape'])
            return_feature = {'image': image, 'image_path': features['image/image_path']}
            return return_feature, label
