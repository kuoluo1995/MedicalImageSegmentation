import numpy as np

from pathlib import Path
from utils import reader_tools


class ImageReader:
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

    def get_data_and_shape(self, idx=None):
        def to_image(i):
            if i is None:
                return self._decode
            if isinstance(i, (int, np.int32, np.int64)):
                return self._decode[i]
            raise ValueError("No supple idx_format: idx_format ({})".format(i))

        if self._decode is None:
            raise ValueError("No data: file_name ({})".format(self.image_path))
        image = to_image(idx)
        if not self._is_label:
            image = image[..., np.newaxis]
            return image.tobytes(), image.shape
        return image.tobytes(), image.shape

    def read(self, image_file):
        self.image_path = str(image_file)
        self.format = Path(image_file).suffix[1:]
        self._reader = eval('reader_tools.' + self.format + '_reader')
        self._writer = eval('reader_tools.' + self.format + '_writer')
        self._decode = self._reader(image_file)
        if self._decode.dtype == float:
            self._decode += 0.5
        self._decode = self._decode.astype(self._type, copy=False)

    def transpose(self, dims):
        self._decode = self._decode.transpose(dims)

    def flipud(self):
        self._decode = np.flipud(self._decode)
