from collections import defaultdict

import numpy as np
import tensorflow as tf
from abc import abstractmethod
from tensorflow.python.framework import errors_impl

import train
from train import metrics_function
from train.config import CustomKeys
from utils.timer import Timer
from utils.reader_tools import VolumeReader
from utils.map_struct import flat_dict_convert_solid_dict


class BaseEvaluate:
    name = None
    eval_steps = 0
    metric_list = None
    estimator = None
    model_dict = None
    dataset = None

    _metric_values = dict()
    save_image = None
    show_each_evaluate = None

    def __init__(self):
        tf.logging.info('#### init {} evaluate'.format(self.name))

    def clear_metrics(self):
        for key in self._metric_values:
            self._metric_values[key].clear()

    def append_metrics(self, params):
        for key, value in params.items():
            if key not in self._metric_values:
                self._metric_values[key] = list()
            self._metric_values[key].append(value)

    def _predict_with_session(self, session, predictions, steps):
        # todo improve predictions
        feed_dict = dict()
        feed_dict[self.estimator.handler] = self.estimator.mode_dict[CustomKeys.EVAL].handler
        try:
            # Initialize evaluation iterator
            session.run(self.estimator.mode_dict[CustomKeys.EVAL].iterator.initializer)
            counter = 0
            while True:
                if steps and counter >= steps:
                    break
                prediction_evaluated = session.run(predictions, feed_dict)
                yield prediction_evaluated
                counter += 1
        except errors_impl.OutOfRangeError:
            pass

    @abstractmethod
    def set_config(self, params):
        pass

    @abstractmethod
    def compare(self, *args, **kwargs):
        pass

    @abstractmethod
    def evaluate_with_session(self, session):
        pass

    @abstractmethod
    def set_eval_config(self, params):
        pass

    @abstractmethod
    def evaluate(self):
        pass
