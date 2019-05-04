import tensorflow as tf

from train.config import CustomKeys


def get_mertrics(mertrics_name, logits, labels, eps):
    return eval(mertrics_name)(logits, labels, eps)


def get_total_mertrics():
    for metric in tf.get_collection(CustomKeys.METRICS):
        yield metric


def dice(logits, labels, eps):
    eps = float(eps)
    dim = len(logits.get_shape())
    sum_axis = list(range(1, dim))
    with tf.variable_scope('Dice', [logits, labels, eps]):
        logits = tf.cast(logits, tf.float32)
        labels = tf.cast(labels, tf.float32)

        AB = tf.reduce_sum(logits * labels, axis=sum_axis)
        A = tf.reduce_sum(logits, axis=sum_axis)
        B = tf.reduce_sum(labels, axis=sum_axis)
        dice = (2 * AB + eps) / (A + B + eps)
        # todo improve tf.identity(dice, name="value")
        dice = tf.reduce_mean(dice, name="value")
        tf.add_to_collection(CustomKeys.METRICS, dice)
        return dice
