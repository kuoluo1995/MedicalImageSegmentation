import numpy as np
import tensorflow as tf


def get_loss(loss_name, logits, label, **kwargs):
    eval(loss_name)(logits, label, **kwargs)


def get_total_loss():
    def _regularization_loss(name='Losses/total_regularization_loss:0'):
        graph = tf.get_default_graph()
        try:
            regularization_loss = graph.get_tensor_by_name(name)
        except KeyError:
            with tf.variable_scope("Losses/"):
                regularization_loss = tf.losses.get_regularization_loss()
        return regularization_loss

    total_losses = tf.losses.get_losses() + [_regularization_loss()]
    for loss in total_losses:
        yield loss


def sparse_softmax_cross_entropy(logits, label, **kwargs):
    tf.logging.info('................>>>>>>>>>>>>>>>> loss: sparse softmax cross entropy')
    loss = tf.losses.sparse_softmax_cross_entropy(label, logits)
    return loss


def weighted_sparse_softmax_cross_entropy(logits, label, classes, **kwargs):
    def _compute_weights():
        batch_size = label.shape[0]
        weights_ = list()
        for class_key, value in classes.items():
            weights_.append(value['weight'])
        weights_ = tf.constant([weights_ for _ in range(batch_size)], dtype=tf.float32)
        weights_ = tf.reduce_sum(weights_[:, None, None, :] * tf.one_hot(label, len(classes)), axis=-1)
        return weights_

    tf.logging.info('................>>>>>>>>>>>>>>>> loss: weighted sparse softmax cross entropy')
    with tf.variable_scope('WeightedSoftmaxCrossEntropy'):
        # todo improve 计算比重
        weights = _compute_weights()
        loss = tf.losses.sparse_softmax_cross_entropy(label, logits, weights)
        return loss
