import numpy as np
from pathlib import Path
from utils import reader_utils


class ImageReader:
    image_path = None
    format = None
    type = np.int16
    _decode = None

    is_label = None
    _reader = None
    _writer = None

    image_channel = None
    header = None

    @property
    def shape(self):
        return self._decode.shape

    def set_config(self, params):
        self.type = params['type']
        self.is_label = params['is_label']
        self.image_channel = params['image_channel']

    def read(self, image_file):
        self.image_path = str(image_file)
        self.format = Path(image_file).suffix[1:]
        self._reader = eval('reader_utils.' + self.format + '_reader')
        self._writer = eval('reader_utils.' + self.format + '_writer')
        self._decode = self._reader(image_file)
        if self._decode.dtype == float:
            self._decode += 0.5
        self._decode = self._decode.astype(self.type, copy=False)

    def set_image(self, image_array):
        self._decode = image_array

    def get_data(self, idx=None):
        def to_image():
            if idx is None:
                return self._decode
            if isinstance(idx, (int, np.int32, np.int64)):
                if self.is_label:  # todo improve 可改善，这里写的有点死了
                    return self._decode[idx]
                else:
                    slices = list()
                    for i in range(self.image_channel):
                        current_index = idx - self.image_channel // 2 + i
                        if current_index < 0 or current_index >= self._decode.shape[0]:
                            slices.append(np.zeros(self._decode.shape[1:], dtype=self._decode.dtype))
                        else:
                            slices.append(self._decode[current_index])
                    return np.stack(slices, axis=-1)

        if self._decode is None:
            raise ValueError('no data: file_name ({})'.format(self.image_path))
        image = to_image()
        return image

    def transpose(self, dims):
        self._decode = self._decode.transpose(dims)

    def flipud(self):
        self._decode = np.flipud(self._decode)

    def read_header_dict(self, image_path):
        self.image_path = str(image_path)
        self.format = Path(image_path).suffix[1:]
        self._reader = eval('reader_utils.' + self.format + '_header_reader')
        self._writer = eval('reader_utils.' + self.format + '_writer')
        self.header = self._reader(image_path)

    def save(self, image_array, image_path):
        self._writer(str(image_path), self.header, image_array)
