import os.path as osp
import re
import numpy as np
import json


def extract_numbers(string):
    numbers = re.findall(r'[-+]?\d+\.\d+', string)
    return numbers


def read_txt(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    velocity = extract_numbers(lines[2])
    location = extract_numbers(lines[4])
    return velocity, location[0:3]


if __name__ == '__main__':
    data_path = 'G:/SynWoodScape_V0.1.1/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/vehicle_data'
    target_path = 'E:/WoodScape-master/omnidet/WoodScape_ICCV19/vehicle_data'
    num_views = 500
    view_list = ['FV', 'MVL', 'MVR', 'RV']
    last_image_num = 8234
    for i in range(num_views):
        present_txt_path = osp.join(data_path, 'rgb_images', str(i).zfill(5)+'.txt')
        previous_txt_path = osp.join(data_path, 'previous_images', str(i).zfill(5) + '.txt')
        curr_v, curr_l = read_txt(present_txt_path)
        prev_v, prev_l = read_txt(previous_txt_path)
        curr_speed = np.sqrt(float(curr_v[0])**2+float(curr_v[1])**2+float(curr_v[2])**2)*3.6
        prev_speed = np.sqrt(float(prev_v[0])**2+float(prev_v[1])**2+float(prev_v[2])**2)*3.6
        displacement = np.sqrt((float(curr_l[0])-float(prev_l[0]))**2+(float(curr_l[1])-float(prev_l[1]))**2
                               +(float(curr_l[2])-float(prev_l[2]))**2)
        t1 = 0
        if curr_speed+prev_speed > 0:
            t1 = displacement/(0.5*(curr_speed+prev_speed)/3.6)*1e6
        t0 = 0
        for j, view in enumerate(view_list):
            target_name = str(i * 4 + last_image_num + j).zfill(5) + '_' + view
            curr_data = {"frame_id": target_name+'.png', "timestamp": str(round(t1)).zfill(8), "ego_speed": "{:.6f}".format(curr_speed)}
            prev_data = {"frame_id": target_name+'.png', "timestamp": str(round(t0)).zfill(8), "ego_speed": "{:.6f}".format(prev_speed)}
            present_json_path = osp.join(target_path, 'rgb_images', target_name+'.json')
            previous_json_path = osp.join(target_path, 'previous_images', target_name + '.json')
            with open(present_json_path, 'w') as file0:
                json.dump(curr_data, file0)
            with open(previous_json_path, 'w') as file1:
                json.dump(prev_data, file1)
