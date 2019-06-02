from tensorboard.plugins.mesh import summary as mesh_summary

...

point_cloud = tf.constant([[[0.19, 0.78, 0.02], ...]], shape=[1, 1064, 3])
point_colors = tf.constant([[[128, 104, 227], ...]], shape=[1, 1064, 3])

summary = mesh_summary.op('point_cloud', vertices=point_cloud, colors=point_colors)