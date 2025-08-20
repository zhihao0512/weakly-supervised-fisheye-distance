import shutil
import os


def copy_and_rename_file(source_path, target_path, new_file_name):
    shutil.copy(source_path, target_path)
    new_file_path = os.path.join(target_path, new_file_name)
    os.rename(os.path.join(target_path, os.path.basename(source_path)), new_file_path)


if __name__ == '__main__':
    data_path = 'G:/SynWoodScape_V0.1.1/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/depth_maps/raw_data'
    target_path = 'G:/SynWoodScape_V0.1.1/SynWoodScape_V0.1.1/SynWoodScape_V0.1.0/depth_maps/raw_data2'
    num_views = 500
    view_list = ['FV', 'MVL', 'MVR', 'RV']
    last_image_num = 8234
    folder_name = 'previous_images'
    for i in range(num_views):
        for j, view in enumerate(view_list):
            source_name = str(i).zfill(5)+'_'+view+'.npy'
            target_name = str(i*4+last_image_num+j).zfill(5)+'_'+view+'.npy'
            source_path = os.path.join(data_path, source_name)
            copy_and_rename_file(source_path, target_path, target_name)
