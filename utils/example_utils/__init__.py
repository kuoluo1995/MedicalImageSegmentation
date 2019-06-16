from utils.example_utils.two_d_example import TwoDExample
from utils.example_utils.three_d_example import ThreeDExample


def create_example(example_name):
    class_instance = eval(example_name)()
    return class_instance
