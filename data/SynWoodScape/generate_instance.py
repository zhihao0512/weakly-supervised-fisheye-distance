import cv2
import os.path as osp
import numpy as np


if __name__ == '__main__':
    data_path = 'G:/SynWoodScape_V0.1.1/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/instance_annotations/gtLabels'
    target_path = 'E:/WoodScape-master/omnidet/WoodScape_ICCV19/segment_annotations/instance'
    num_views = 500
    view_list = ['FV', 'MVL', 'MVR', 'RV']
    last_image_num = 8234
    for i in range(num_views):
        for j, view in enumerate(view_list):
            source_name = str(i).zfill(5) + '_' + view
            target_name = str(i * 4 + last_image_num + j).zfill(5) + '_' + view
            source_label = cv2.imread(osp.join(data_path, source_name+'.png'), 0)
            target_label = source_label+1
            cv2.imwrite(osp.join(target_path, target_name+'.png'), target_label)

                
