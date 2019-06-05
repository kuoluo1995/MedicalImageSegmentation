import tensorflow as tf

from model_component.config import CustomKeys
from model_component.solver import learning_rate_funcation, optimizer_funcation


class Solver:
    slow_start_step = 0
    slow_start_learning_rate = 1e-4

    _learning_rate = None
    _optimizer = None

    def __init__(self):
        tf.logging.info('#### init solver')

    def set_config(self, params):
        tf.logging.info('..... setting solver config')
        self.slow_start_step = params['slow_start_step']
        self.slow_start_learning_rate = params['slow_start_learning_rate']
        self._learning_rate = params['learning_rate']
        self._optimizer = params['optimizer']

    def get_train_optimizer(self, loss):
        tf.logging.info('................ get solver model_component optimizer')
        self._learning_rate = learning_rate_funcation.get_learning_rate(slow_start_step=self.slow_start_step,
                                                                        slow_start_learning_rate=self.slow_start_learning_rate,
                                                                        **self._learning_rate)
        self._optimizer = optimizer_funcation.get_model_optimizer(self._optimizer['name'],
                                                                  self._learning_rate,
                                                                  self._optimizer)

        with tf.variable_scope('Optimizer'):
            update_ops = tf.get_collection(CustomKeys.UPDATE_OPS)
            if update_ops:
                with tf.control_dependencies(update_ops):
                    train_optimizer = self._optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
            else:
                train_optimizer = self._optimizer.minimize(loss, global_step=tf.train.get_or_create_global_step())
            return train_optimizer
