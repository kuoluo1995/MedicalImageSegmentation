import yaml
import build_dataset


def main():
    # 创建训练数据
    config_file = 'carcass_dataset_channel_1'
    config_file = open('config/build_dataset/' + config_file + '.yaml', 'rb')
    dataset_config = yaml.load(config_file)
    dataset = build_dataset.create_dataset(dataset_config['dataset']['name'])
    dataset.set_build_config(dataset_config)
    dataset.build_dataset()


if __name__ == "__main__":
    main()
