import os
import numpy as np

image_FNAME = 'image_images.npy'
NUM_imageS_PER_FOLDER = 20

def validation_data_for_folders_and_save(folders_list, out_path):
    
    valid_data = get_validation_data_for_folders(folders_list)
    np.save(out_path, valid_data)

def get_validation_data_for_folders(folders_list):
    
    images = list()
    stl_z = list()
    for fold in folders_list:
        hm, sz = create_validation_data_for_folder(fold)
        images += hm
        stl_z += sz
    return {'images': np.array(images), 'stl_z': np.array(stl_z)}

def create_validation_data_for_folder(folder):
    
    images_dat = np.load(os.path.join(folder, image_FNAME)).tolist()
    dk = [key for key, value in images_dat.items() if 'height_mat_mm' in value]
    order = np.random.permutation(len(dk))
    num_maps = min(NUM_imageS_PER_FOLDER, len(order))
    images = list()
    stl_z = list()
    for index in order[:num_maps]:
        images.append(images_dat[dk[index]]['height_mat_mm'])
        stl_z.append(images_dat[dk[index]]['stl_z'])
    return images, stl_z