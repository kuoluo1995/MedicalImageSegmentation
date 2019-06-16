import yaml


def read(path):
    with path.open('r') as file:
        k_folds = yaml.load(file)
    return k_folds


def write(path, data):
    with path.open('w') as file:
        yaml.dump(data, file)
