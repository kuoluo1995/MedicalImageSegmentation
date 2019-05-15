import numpy as np
import tensorflow as tf
from medpy import metric as metric_lib

from train.config import CustomKeys


def get_mertrics(mertrics_name, logits, labels, eps, name):
    return eval(mertrics_name)(logits, labels, eps, name)


def get_total_mertrics():
    for metric in tf.get_collection(CustomKeys.METRICS):
        yield metric


def dice(logits, labels, eps, name):
    eps = float(eps)
    dim = len(logits.get_shape())
    sum_axis = list(range(1, dim))
    with tf.variable_scope(name, [logits, labels, eps]):
        logits = tf.cast(logits, tf.float32)
        label = tf.cast(labels, tf.float32)

        AB = tf.reduce_sum(logits * label, axis=sum_axis)
        A = tf.reduce_sum(logits, axis=sum_axis)
        B = tf.reduce_sum(label, axis=sum_axis)
        dice = (2 * AB + eps) / (A + B + eps)
        # todo improve tf.identity(dice, name="value")
        dice = tf.reduce_mean(dice, name="value")
        tf.add_to_collection(CustomKeys.METRICS, dice)
        return dice


def metric_3d(logits3d, labels3d, required=None, **kwargs):
    """
    Compute 3D metrics:
    * (Dice) Dice Coefficient
    * (VOE)  Volumetric Overlap Error
    * (VD)   Relative Volume Difference
    * (ASD)  Average Symmetric Surface Distance
    * (RMSD) Root Mean Square Symmetric Surface Distance
    * (MSD)  Maximum Symmetric Surface Distance

    Parameters
    ----------
    logits3d: ndarray
        3D binary prediction, shape is the same with `labels3d`, it should be an int array or boolean array.
    labels3d: ndarray
        3D labels for segmentation, shape [None, None, None], it should be an int array
        or boolean array. If the dimensions of `logits3d` and `labels3d` are greater than
        3, then `np.squeeze` will be applied to remove extra single dimension and then
        please make sure these two variables are still have 3 dimensions. For example,
        shape [None, None, None, 1] or [1, None, None, None, 1] are allowed.
    required: str or list
        a string or a list of string to specify which metrics need to be return, default
        this function will return all the metrics listed above. For example, if use
        ```python
        _metric_3D(logits3D, labels3D, require=["Dice", "VOE", "ASD"])
        ```
        then only these three metrics will be returned.
    kwargs: dict
        sampling: list
            the pixel resolution or pixel size. This is entered as an n-vector where n
            is equal to the number of dimensions in the segmentation i.e. 2D or 3D. The
            default value is 1 which means pixls are 1x1x1 mm in size

    Returns
    -------
    metrics required

    Notes
    -----
    Thanks to the code snippet from @MLNotebook's blog.

    [Blog link](https://mlnotebook.github.io/post/surface-distance-function/).
    """
    metrics = ["Dice", "VOE", "RVD", "ASSD", "RMSD", "MSD"]
    need_dist_map = False

    if required is None:
        required = metrics
    elif isinstance(required, str):
        required = [required]
        if required[0] not in metrics:
            raise ValueError("Not supported metric: %s" % required[0])
        elif required in metrics[3:]:
            need_dist_map = True
        else:
            need_dist_map = False

    for req in required:
        if req not in metrics:
            raise ValueError("Not supported metric: %s" % req)
        if (not need_dist_map) and req in metrics[3:]:
            need_dist_map = True

    if logits3d.ndim > 3:
        logits3d = np.squeeze(logits3d)
    if labels3d.ndim > 3:
        labels3d = np.squeeze(labels3d)

    assert logits3d.shape == labels3d.shape, ("Shape mismatch of logits3D and labels3D. \n"
                                              "Logits3D has shape %r while labels3D has "
                                              "shape %r" % (logits3d.shape, labels3d.shape))
    logits3d = logits3d.astype(np.bool)
    labels3d = labels3d.astype(np.bool)

    metrics_3d = {}
    sampling = kwargs.get("sampling", [1., 1., 1.])

    if need_dist_map:
        from utils.surface import Surface
        if np.count_nonzero(logits3d) == 0 or np.count_nonzero(labels3d) == 0:
            metrics_3d['ASSD'] = 0
            metrics_3d['MSD'] = 0
        else:
            eval_surf = Surface(logits3d, labels3d, physical_voxel_spacing=sampling,
                                mask_offset=[0., 0., 0.],
                                reference_offset=[0., 0., 0.])

            if "ASSD" in required:
                metrics_3d["ASSD"] = eval_surf.get_average_symmetric_surface_distance()
                required.remove("ASSD")
            if "MSD" in required:
                metrics_3d["MSD"] = eval_surf.get_maximum_symmetric_surface_distance()
            if "RMSD" in required:
                metrics_3d["RMSD"] = eval_surf.get_root_mean_square_symmetric_surface_distance()

    if required:
        if "Dice" in required:
            metrics_3d["Dice"] = metric_lib.dc(logits3d, labels3d)
        if "VOE" in required:
            metrics_3d["VOE"] = 1. - metric_lib.jc(logits3d, labels3d)
        if "RVD" in required:
            metrics_3d["RVD"] = metric_lib.ravd(logits3d, labels3d)

    return metrics_3d
