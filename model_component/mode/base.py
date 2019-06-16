import tensorflow as tf
from abc import abstractmethod
from functools import partial
from tensorflow.contrib import data as contrib_data
from tensorflow.python import ops
from tensorflow.python.data import Dataset, experimental, TFRecordDataset
from tensorflow.python.estimator import util
from model_component.config import CustomKeys
from utils import example_utils, image_process_utils, yaml_utils


class BaseMode:
    name = None
    dataset_path = None
    example = None
    dataset_dict = None
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
    def adjust_window_size(self, image, min_window, max_window):
        pass

    @abstractmethod
    def get_extra_feature(self, **params):
        pass
