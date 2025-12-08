import yaml

from configs import eval


def load_config_file(config_file, **kwargs):
    with open(config_file) as fh:
        config_dict = yaml.safe_load(fh)
    config_dict = {**config_dict, **kwargs}
    return config_dict

