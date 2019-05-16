import tensorflow as tf


def get_loss(loss_name, logits, label, classes):
    eval(loss_name)(logits, label, classes)


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


def weighted_sparse_softmax_cross_entropy(logits, label, classes):
    tf.logging.info('................>>>>>>>>>>>>>>>> loss: weighted sparse softmax cross entropy')
    with tf.variable_scope('WeightedSoftmaxCrossEntropy'):
        # todo improve 计算比重
        loss = tf.losses.sparse_softmax_cross_entropy(label, logits)
        return loss
