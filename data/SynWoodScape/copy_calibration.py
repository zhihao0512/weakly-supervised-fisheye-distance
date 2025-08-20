import os.path as osp
import json


def modify_key(dict, old_key, new_key):
    new_dict = {}
    for key, value in dict.items():
        if key == "extrinsic":
            new_dict2 = {}
            for key2, value2 in value.items():
                if key2 == old_key:
                    new_dict2[new_key] = value2
                else:
                    new_dict2[key2] = value2
            new_dict["extrinsic"] = new_dict2
        else:
            new_dict[key] = value
    return new_dict


if __name__ == '__main__':
    data_path = 'G:/SynWoodScape_V0.1.1/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/calibration_data'
    target_path = 'E:/WoodScape-master/omnidet/WoodScape_ICCV19/calibration_data'
    num_views = 500
    view_list = ['FV', 'MVL', 'MVR', 'RV']
    last_image_num = 8234
    json_data = []
    for j, view in enumerate(view_list):
        file_path = osp.join(data_path, view+'.json')
        with open(file_path, 'r') as file:
            data = json.load(file)
            data = modify_key(data, "translation used in CARLA (CARLA reference)", "translation")
            json_data.append(data)
    for i in range(num_views):
        for j, view in enumerate(view_list):
            target_name = str(i * 4 + last_image_num + j).zfill(5) + '_' + view
            target_path2 = osp.join(target_path, target_name+'.json')
            with open(target_path2, 'w') as file0:
                json.dump(json_data[j], file0)