from model_component.networks.UNet import UNet


def create_network(network_name):
    class_instance = eval(network_name)()
    return class_instance
