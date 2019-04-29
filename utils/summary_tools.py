import tensorflow as tf


def scalar(writer, iteration, tags, values):
    summary = []
    for tag, value in zip(tags, values):
        summary.append(tf.Summary.Value(tag=tag, simple_value=value))
    writer.add_summary(tf.Summary(value=summary), int(iteration))
