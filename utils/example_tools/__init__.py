from utils.example_tools.two_d_example import TwoDExample
from utils.example_tools.three_d_example import ThreeDExample


def create_example(example_name, writer=None, image_tool=None, label_tool=None):
    class_instance = eval(example_name)()
    class_instance.writer = writer
    class_instance.image_tool = image_tool
    class_instance.label_tool = label_tool
    return class_instance
