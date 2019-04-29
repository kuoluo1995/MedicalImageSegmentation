import tensorflow as tf


def get_model_optimizer(name, learning_rate, params):
    return eval(name)(learning_rate, params)


def adam(learning_rate, params):
    optimizer = tf.train.AdamOptimizer(learning_rate, **params)
    tf.logging.info('set completed optimizer:Adam')
    return optimizer
