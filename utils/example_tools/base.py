import tensorflow as tf
from abc import abstractmethod
from collections import Iterable
from utils import image_process_operations


def feature_to_int64_list(value):
    if not isinstance(value, Iterable):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def feature_to_bytes_list(source_value):
    def to_bytes(value):
        if isinstance(value, str):
            return value.encode()
        if isinstance(value, bytes):
            return value
        return TypeError('Only str and bytes are supported, got {}'.format(type(value)))

    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[to_bytes(source_value)]))


class BaseExample:
    name = None
    writer = None
    image_reader = None
    label_reader = None
    dataset_class = None

    @abstractmethod
    def _create_example_format(self):
        pass

    def write_example(self, i, fold):
        for j, data in enumerate(fold):
            print("Converting {}: fold {}, {}/{}".format(str(self.__class__), i + 1, j + 1, len(fold)))
            self.image_reader.read(data['image'])
            self.label_reader.read(data['label'])

            if self.image_reader.shape != self.label_reader.shape:
                raise RuntimeError("Shape mismatched between image and label: {} vs {},image_path:{} ".format(
                    self.image_reader.shape, self.label_reader.shape, self.image_reader.image_path))

            for example in self._create_example_format():
                self.writer.write(example.SerializeToString())

    @abstractmethod
    def read_example(self, example_proto, mode, **params):
        pass
