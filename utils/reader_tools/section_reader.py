import numpy as np

from pathlib import Path
from utils.mhd_tools import *


class ImageReader:
    name = None
    image_path = None
    format = None
    _type = None
    _decode = None

    _is_label = None
    _reader = None
    _writer = None

    @property
    def shape(self):
        return self._decode.shape

    def __init__(self, image_type=np.int16, is_label=False):
        self._type = image_type
        self._is_label = is_label

    def get_data_and_shape(self, idx=None, image_channel=None):
        def to_image():
            if idx is None:
                return self._decode
            if isinstance(idx, (int, np.int32, np.int64)):
                slices = list()
                for i in range(image_channel):
                    if idx - image_channel // 2 + i < 0 or idx - image_channel // 2 + i >= self._decode.shape[0]:
                        slices.append(np.zeros(self._decode.shape[1:], dtype=self._decode.dtype))
                    else:
                        slices.append(self._decode[idx - image_channel // 2 + i])
                if self._is_label:
                    return np.stack(slices, axis=-1)
                else:
                    return np.concatenate(slices, axis=-1)

        if self._decode is None:
            raise ValueError("No data: file_name ({})".format(self.name))
        image = to_image()
        return image.tobytes(), image.shape

    def read(self, image_file):
        self.name = Path(image_file).stem
        self.image_path = str(image_file)
        self.format = Path(image_file).suffix[1:]
        self._reader = eval(self.format + '_reader')
        self._writer = eval(self.format + '_writer')
        self._decode = self._reader(image_file).astype(self._type, copy=False)

    def transpose(self, dims):
        self._decode = self._decode.transpose(dims)

    def flipud(self):
        self._decode = np.flipud(self._decode)

    def extract_region(self, align, padding, min_bbox_shape=None):
        mask = np.asarray(self._decode, np.bool)
        ndim = mask.ndim
        align = np.array(align, dtype=np.int32)

        pass
