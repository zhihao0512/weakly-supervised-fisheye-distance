import cv2
import os.path as osp
import numpy as np


if __name__ == '__main__':
    data_path = 'G:/SynWoodScape_V0.1.1/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/semantic_annotations/gtLabels'
    target_path = 'E:/WoodScape-master/omnidet/WoodScape_ICCV19/segment_annotations/semantic'
    num_views = 500
    view_list = ['FV', 'MVL', 'MVR', 'RV']
    last_image_num = 8234
    for i in range(num_views):
        for j, view in enumerate(view_list):
            source_name = str(i).zfill(5) + '_' + view
            target_name = str(i * 4 + last_image_num + j).zfill(5) + '_' + view
            source_label = cv2.imread(osp.join(data_path, source_name+'.png'), 0)
            target_label = np.zeros_like(source_label)
            target_label[source_label == 24] = 1
            target_label[source_label == 13] = 2
            target_label[(source_label == 7) | (source_label == 8)] = 3
            target_label[source_label == 1] = 4
            target_label[(source_label == 10) | (source_label == 21)] = 5
            target_label[source_label == 4] = 6
            target_label[(source_label == 5) | (source_label == 3) | (source_label == 22) | (source_label == 12)
                         | (source_label == 18) | (source_label == 2) | (source_label == 11) | (source_label == 16)] = 7
            target_label[source_label == 6] = 8
            target_label[(source_label == 9) | (source_label == 15) | (source_label == 19)] = 9
            target_label[(source_label == 14) | (source_label == 20)] = 10
            cv2.imwrite(osp.join(target_path, target_name+'.png'), target_label)

                
