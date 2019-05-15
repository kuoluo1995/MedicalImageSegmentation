from utils.reader_tools.image_reader import *


class VolumeReader(ImageReader):
    bbox = list()

    def bounding_box(self, align, padding, min_bbox):
        def _bounding_box_mask():
            mask = np.asarray(self._decode, np.bool)
            if np.count_nonzero(mask) == 0:
                return np.zeros(shape=(mask.ndim * 2))
            mask_values = np.array([1]).reshape(-1, 1)
            for d in reversed(range(mask.ndim)):
                pass

        ndim = self._decode.ndim
        align = np.array(align, dtype=np.int32)
        min_bbox = np.array(min_bbox, dtype=np.int32)
        image_shape = np.array(self._decode.shape)
        pre_bbox = _bounding_box_mask()
        pass
