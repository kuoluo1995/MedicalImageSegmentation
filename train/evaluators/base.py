from collections import defaultdict

import numpy as np
import tensorflow as tf
from abc import abstractmethod
from tensorflow.python.framework import errors_impl


class BaseEvaluate:
    name = None
    eval_steps = 0
    metric_list = None
    estimator = None
    model_dict = None

    _metric_values = defaultdict(list)

    def __init__(self):
        tf.logging.info('init {} evaluate'.format(self.name))

    def clear_metrics(self):
        for key in self._metric_values:
            self._metric_values[key].clear()

    def append_metrics(self, params):
        for key, value in params.items():
            self._metric_values[key].append(value)

    def _predict_with_session(self, session, prediction_evaluated, steps):
        def _get_batch_length():
            batch_length = None
            for key, value in prediction_evaluated.items():
                batch_length = batch_length or value.shape[0]
            return batch_length

        # todo improve predictions
        feed_dict = dict()
        feed_dict[self.estimator.handler] = self.estimator.mode_dict['EvalMode'].handler
        try:
            # Initialize evaluation iterator
            session.run(self.estimator.mode_dict['EvalMode'].iterator.initializer)
            counter = 0
            while True:
                if steps and counter >= steps:
                    break
                prediction_evaluated = session.run(prediction_evaluated, feed_dict)
                for i in range(_get_batch_length()):
                    yield {key: value[i] for key, value in prediction_evaluated.items()}
                counter += 1
        except errors_impl.OutOfRangeError:
            tf.logging.error("predict_with_session error")
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
