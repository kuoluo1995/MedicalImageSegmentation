from train.config import CustomKeys


def flat_dict_convert_solid_dict(flat_dict):
    def list_convert_dict(temp_keys, temp_dict):
        if len(temp_keys) == 1:
            temp_dict.setdefault(temp_keys[0], value)
            return temp_dict
        temp_dict.setdefault(temp_keys[0], list_convert_dict(temp_keys[1:], temp_dict.get(temp_keys[0], dict())))
        return temp_dict

    return_dict = dict()
    for key_string, value in flat_dict.items():
        keys = key_string.split(CustomKeys.SEPARATOR)
        list_convert_dict(keys, return_dict)
    return return_dict
