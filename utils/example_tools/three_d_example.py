from utils.example_tools.base import *


class ThreeDExample(BaseExample):
    name = 'threeD'
    extra_feature = {'padding_len': {CustomKeys.VALUE: 0, CustomKeys.TYPE: tf.int64}}

    def _create_example_format(self):
        self.dataset_class.deal_image(self.image_reader)
        self.dataset_class.deal_image(self.label_reader)
        image, image_shape = self.image_reader.get_data_and_shape()
        label, label_shape = self.label_reader.get_data_and_shape()
        feature = {
            'image/image_path': feature_to_bytes_list(self.image_reader.image_path),
            'image/shape': feature_to_int64_list(image_shape),
            'image/encoded': feature_to_bytes_list(image),
            'segmentation/shape': feature_to_int64_list(label_shape),
            'segmentation/encoded': feature_to_bytes_list(label),
        }
        yield tf.train.Example(features=tf.train.Features(feature=feature))

    def read_example(self, example_proto, mode, **params):
        def padding(source_image, shape):
            padding_shape = tf.concat(([padding_len], shape[1:]), axis=0)
            padding_image = tf.zeros(padding_shape, dtype=source_image.dtype)
            new_image = tf.concat((source_image, padding_image), axis=0)
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
            padding_len = params['batch_size'] - features['image/shape'][0] % params['batch_size']
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
            feature = {CustomKeys.IMAGE: image, CustomKeys.IMAGE_PATH: features['image/image_path'],
                       'padding_len': padding_len}
            for other, value in params['extra_feature'].items():
                if other not in feature:
                    feature[other] = tf.constant(value[CustomKeys.VALUE], dtype=value[CustomKeys.TYPE])
            return feature, label
