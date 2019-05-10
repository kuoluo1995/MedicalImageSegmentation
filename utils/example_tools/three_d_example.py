import tensorflow as tf

from utils.example_tools import base


class ThreeDExample(base.BaseExample):
    def _create_example_format(self):
        image, image_shape = self.image_reader.get_data_and_shape()
        label, label_shape = self.label_reader.get_data_and_shape()
        feature = {
            'image/name': base.feature_to_bytes_list(self.image_reader.name),
            'image/shape': base.feature_to_int64_list(image_shape),
            'image/encoded': base.feature_to_bytes_list(image),
            'segmentation/shape': base.feature_to_int64_list(label_shape),
            'segmentation/encoded': base.feature_to_bytes_list(label),
        }
        yield tf.train.Example(features=tf.train.Features(feature=feature))

    def read_example(self, example_proto, mode, **params):
        pass
