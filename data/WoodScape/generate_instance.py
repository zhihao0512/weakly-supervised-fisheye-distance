import os
from glob import glob
import json
import cv2
import numpy as np


def edge_detection(label):
    label0 = np.zeros([label.shape[0]+1, label.shape[1]+1], dtype=label.dtype)
    label0[:-1, :-1] = label
    edge_x = label0[:-1, :-1]-label0[:-1, 1:]
    edge_y = label0[:-1, :-1]-label0[1:, :-1]
    edge = np.zeros(edge_x.shape, dtype=np.uint8)
    edge[np.bitwise_or(edge_x != 0, edge_y != 0)] = 255
    return edge


def check_label_categories(class_names, label):
    for idx, c in enumerate(class_names):
        if label in c:
            return idx
    return -1


def draw_label_map(anno, class_names, w, h):
    instance_label = np.zeros([h, w], dtype=np.int16)
    semantic_label = np.zeros([h, w], dtype=np.uint8)
    for idx, info in enumerate(anno):
        poly = np.array(info['segmentation']).astype(np.int32)
        class_label = info['tags'][0]
        instance_label = cv2.fillPoly(instance_label, [poly], idx+1)
        semantic_label = cv2.fillPoly(semantic_label, [poly], check_label_categories(class_names, class_label)+1)
    return instance_label, semantic_label


if __name__ == '__main__':
    data_path = 'E:/WoodScape-master/omnidet/WoodScape_ICCV19'
    instance_annos = sorted(glob(os.path.join(data_path, 'instance_annotations', '*.json')))
    class_names = [["green_strip", "ego_vehicle"], ["sky"], ["road_surface", "curb", "free_space"], ["construction"],
                   ["bus", "car", "truck", "van", "bicycle", "motorcycle", "train/tram", "trailer", "caravan",
                    "other_wheeled_transport", "grouped_vehicles"],
                   ["movable_object", "person", "rider", "animal",
                   "grouped_pedestrian_and_animals", "grouped_animals"],
                   ["fence", "pole", "traffic_sign", "trafficsign_indistingushable",
                   "unknown_traffic_light", "traffic_light_red", "traffic_light_green", "traffic_light_yellow"],
                   ["lane_marking", "parking_line", "parking_marking", "zebra_crossing",
                   "grouped_botts_dots", "cats_eyes_and_botts_dots", "other_ground_marking"],
                   ["nature"], ["void"]]
    for idx, anno_path in enumerate(instance_annos):
        anno_name = anno_path.split('\\')[-1]
        frame_index, cam_side = anno_name.split('.')[0].split('_')
        data = json.load(open(anno_path))
        instance_anno = data[anno_name]['annotation']
        #class_names = data[anno_name]['annotation-tags']
        image_height = data[anno_name]['image_height']
        image_width = data[anno_name]['image_width']
        instance_label, semantic_label = draw_label_map(instance_anno, class_names, image_width, image_height)
        edge_label = edge_detection(instance_label)
        #cv2.imwrite(os.path.join(data_path, 'segment_annotations/instance', f"{frame_index}_{cam_side}.png"), instance_label)
        cv2.imwrite(os.path.join(data_path, 'segment_annotations/semantic', f"{frame_index}_{cam_side}.png"), semantic_label)
        #cv2.imwrite(os.path.join(data_path, 'segment_annotations/edge', f"{frame_index}_{cam_side}.png"), edge_label)
        if len(instance_anno) > 255:
            print(idx, len(instance_anno))
