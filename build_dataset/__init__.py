from build_dataset.body_dataset import BodyDataset


def create_dataset(dataset_name):
    class_instance = eval(dataset_name)()
    return class_instance
