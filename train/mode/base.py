import tensorflow as tf
from abc import abstractmethod
from functools import partial
from tensorflow.contrib import data as contrib_data
from tensorflow.python import ops
from tensorflow.python.data import Dataset, experimental, TFRecordDataset
from tensorflow.python.estimator import util
from utils import example_tools, image_process_operations, yaml_tools


class BaseMode:
    name = None
    dataset_path = None
    # IteratorStringHandleHook 专用
    handler = None
    iterator = None
    string_handle = None

    def __init__(self):
        tf.logging.info('#### init {} mode'.format(self.name))

    @abstractmethod
    def set_config(self, **params):
        pass

    @abstractmethod
    def get_dataset(self, **params):
        pass

    @abstractmethod
    def init_iterator(self, dataset):
        pass

    @abstractmethod
    def adjust_window_size(self, image, min_window, max_window):
        pass
