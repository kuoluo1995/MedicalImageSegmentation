import tensorflow as tf


def process_image(function_name, **params):
    return eval(function_name)(**params)


def adjust_window_size(image, min_window, max_window, **kwargs):
    with tf.name_scope('Adjust_Window_Size/'):
        image = tf.convert_to_tensor(image)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = (tf.clip_by_value(image, min_window, max_window) - min_window) / (max_window - min_window)
        tf.logging.info('adjust window size')
        return image


def random_zoom_in(image, label, max_scale, seed, **kwargs):
    with tf.name_scope('Random_Zoom_In/'):
        scale = tf.random_uniform([2], 1, max_scale, seed=seed)
        image_shape = tf.shape(image)
        new_size = tf.to_int32(tf.to_float(image_shape[:-1]) * scale)
        expanded_image = tf.expand_dims(image, axis=0)
        central_zoom_in_image = tf.image.resize_bilinear(expanded_image, new_size)
        central_zoom_in_image = tf.squeeze(central_zoom_in_image, axis=0)

        extended_label = tf.expand_dims(tf.expand_dims(label, 0), -1)
        central_zoom_in_label = tf.image.resize_nearest_neighbor(extended_label, new_size)
        central_zoom_in_label = tf.to_float(tf.squeeze(central_zoom_in_label, axis=0))

        combined = tf.concat(axis=-1, values=[central_zoom_in_image, central_zoom_in_label])
        new_shape = tf.concat([image_shape[:-1], [image_shape[-1] + 1]], axis=0)
        cropped_combined = tf.image.random_crop(combined, new_shape, seed=seed)
        cropped_image = cropped_combined[..., :-1]
        cropped_label = tf.to_int32(cropped_combined[..., -1])
        tf.logging.info('augmentation:random zoom in')
        return cropped_image, cropped_label


def random_horizontally_flip(image, label, seed, **kwargs):
    with tf.name_scope('random_horizontally_flip/'):
        label_new = tf.expand_dims(tf.cast(label, image.dtype), axis=-1)
        combined = tf.concat((image, label_new), axis=-1)
        combined_flipped = tf.image.random_flip_left_right(combined, seed)
        tf.logging.info('augmentation:random horizontally flip')
        return combined_flipped[..., :-1], tf.cast(combined_flipped[..., -1], dtype=label.dtype)


def random_noise(image, scale, seed, **kwargs):
    with tf.name_scope('random_noise/'):
        scale = tf.abs(scale)
        noise_mask = tf.random_uniform(tf.shape(image), -scale, scale, seed=seed, dtype=image.dtype)
        new_image = tf.add(image, noise_mask, 'NoisedImage')
        tf.logging.info('augmentation:random noise')
        return new_image, None


def resize_image(image, height, width):
    with tf.name_scope('Resize_Image/'):
        tf.logging.info('resize image')
        return tf.image.resize_bilinear(tf.expand_dims(image, axis=0), [height, width])
