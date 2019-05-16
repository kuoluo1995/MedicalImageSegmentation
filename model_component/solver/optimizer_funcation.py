import tensorflow as tf


def get_model_optimizer(name, learning_rate, params):
    return eval(name)(learning_rate, params)


def adam(learning_rate, params):
    with tf.variable_scope('Solver'):
        tf.logging.info('................>>>>>>>>>>>>>>>> optimizer:Adam')
        optimizer = tf.train.AdamOptimizer(learning_rate, **params)
        return optimizer
