import tensorflow as tf

from train.config import CustomKeys


def get_learning_rate(name, **params):
    return eval(name)(**params)  # todo test 看下值对比下


def piecewise_constant(slow_start_step, slow_start_learning_rate, **params):
    with tf.variable_scope('Solver'):
        tf.logging.info('................>>>>>>>>>>>>>>>> learning rate:piecewise_constant')
        global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.piecewise_constant(global_step, params['boundaries'], params['values'])
        if slow_start_step > 0:
            learning_rate = tf.where(global_step < slow_start_step, slow_start_learning_rate, learning_rate)
        tf.add_to_collection(CustomKeys.LEARNING_RATE, learning_rate)
        return learning_rate
