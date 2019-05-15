from build_dataset.body_dataset import BodyDataset
from build_dataset.carcass_dataset import CarcassDataset


def create_dataset(dataset_name):
    class_instance = eval(dataset_name)()
    return class_instance
