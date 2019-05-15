from utils.example_tools.two_d_example import TwoDExample
from utils.example_tools.three_d_example import ThreeDExample


def create_example(example_name, writer=None, image_reader=None, label_reader=None, dataset_class=None):
    class_instance = eval(example_name)()
    class_instance.writer = writer
    class_instance.image_reader = image_reader
    class_instance.label_reader = label_reader
    class_instance.dataset_class = dataset_class
    return class_instance
