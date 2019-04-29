import tensorflow as tf


def get_loss(loss_name, logits, label, classes):
    eval(loss_name)(logits, label, classes)


def get_total_loss():
    graph = tf.get_default_graph()
    total_losses = tf.losses.get_losses() + [_regularization_loss(graph)]
    for loss in total_losses:
        yield loss


def _regularization_loss(graph, name='Losses/total_regularization_loss:0'):
    try:
        regularization_loss = graph.get_tensor_by_name(name)
    except KeyError:
        with tf.name_scope("Losses/"):
            regularization_loss = tf.losses.get_regularization_loss()
    return regularization_loss


def weighted_sparse_softmax_cross_entropy(logits, label, classes):
    # with tf.name_scope("LabelProcess"):
    #     # todo improve 提供weight
    #     num_classes = logits.shape[-1]
    #     one_hot_labels = tf.one_hot(label, num_classes)
    with tf.name_scope('WeightedSoftmaxCrossEntropy/'):
        # todo improve 计算比重
        loss = tf.losses.sparse_softmax_cross_entropy(label, logits)
        tf.logging.info('loss:weighted sparse softmax cross entropy')
        return loss
