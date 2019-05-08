from utils.reader_tools.section_reader import ImageReader
from utils.reader_tools.volume_reader import VolumeReader


def create_reader(reader_name, image_type, is_label):
    class_instance = eval(reader_name)(image_type, is_label)
    return class_instance
