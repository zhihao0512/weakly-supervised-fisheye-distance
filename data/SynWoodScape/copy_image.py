import cv2
import os.path as osp


if __name__ == '__main__':
    data_path = 'G:/SynWoodScape_V0.1.1/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0'
    target_path = 'E:/WoodScape-master/omnidet/WoodScape_ICCV19'
    num_views = 500
    view_list = ['FV', 'MVL', 'MVR', 'RV']
    last_image_num = 8234
    folder_name = 'previous_images'
    for i in range(num_views):
        for j, view in enumerate(view_list):
            source_name = str(i).zfill(5)+'_'+view
            target_name = str(i*4+last_image_num+j).zfill(5)+'_'+view
            source_image = cv2.imread(osp.join(data_path, 'flow_annotations/rgbLabels', source_name+'.png'))
            cv2.imwrite(osp.join(target_path, 'flow_annotations_flowformersintel/rgbLabels', target_name+'.png'), source_image)
