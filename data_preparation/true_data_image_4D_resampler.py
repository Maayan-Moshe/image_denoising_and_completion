
import numpy as np
from image_3d_sampler import Image3Dsampler
from image_4d_to_height_map_resampler import Image4DToHeightMapResampler

SHAPE = (155, 208)
FILES_NAMES = {'in_img4D': '4d_images.npy', 'stl': 'lower.stl', 
               'out_img5d': '5d_images.npy', 'out_matrices': '5d_matrices.npy',
               'out_image': 'image_images.npy'}
               
ROTAION_APP_TO_MOD = np.eye(4) #np.array(((0,1,0,0), (-1,0,0,0), (0,0,1,0), (0,0,0,1)))
               
def resample_height_map_and_add_true_z(folder, fnames = FILES_NAMES, max_z_mm = 25):
    import os
    
    images_4D_dat_path = os.path.join(folder, fnames['in_img4D'])
    stl_path = os.path.join(folder, fnames['stl'])
    out_5d_path = os.path.join(folder, fnames['out_image'])
    resample_add_true_z_values_and_save(images_4D_dat_path, stl_path, out_5d_path, max_z_mm)
    
def convert_save_image_images_to_matrices(folder,
                                         images_fname = 'image_images.npy',
                                         matirces_fname = 'image_matrices.npy'):
    from .data_preparation import prepare_image_data
    import os
    himgs_path = os.path.join(folder, images_fname)
    hmatrices_data = prepare_image_data(himgs_path)
    hmatrices_path = os.path.join(folder, matirces_fname)
    np.save(hmatrices_path, hmatrices_data)
               
def add_true_z_values_and_save_folder(folder, fnames = FILES_NAMES, max_z_mm = 25):
    import os
    
    images_4D_dat_path = os.path.join(folder, fnames['in_img4D'])
    stl_path = os.path.join(folder, fnames['stl'])
    out_5d_path = os.path.join(folder, fnames['out_img5d'])
    out_ma_path = os.path.join(folder, fnames['out_matrices'])
    add_true_z_values_and_save(images_4D_dat_path, stl_path, out_5d_path, max_z_mm)
    convert_to_matrix_and_save(out_5d_path, out_ma_path)
    
def convert_to_matrix_and_save(in_5d_path, out_ma_path):
    from .data_preparation import prepare_scan_data
    
    mat_data = prepare_scan_data(in_5d_path)
    np.save(out_ma_path, mat_data)  
    
def resample_add_true_z_values_and_save(images_4D_dat_path, stl_path, out_path, max_z_mm):
    
    resampler = Image4DToHeightMapResampler()
    sampler = Image3Dsampler(stl_path, max_z_mm, ROTAION_APP_TO_MOD)
    images_4D_dat = np.load(images_4D_dat_path).tolist()
    for img_dat in images_4D_dat.itervalues():
        add_height_map_and_true_z(img_dat, sampler, resampler)
    np.save(out_path, images_4D_dat)
    
def add_height_map_and_true_z(img_dat, sampler, resampler):
    
    if 'scn_to_wrld_tx' not in img_dat:
        print(str(img_dat['scan_hardware_id']) + ' has no scn_to_wrld_tx')
        return
    XY_mat_mm = resampler.get_xy_matrices()
    true_Z_mat_mm = sampler.get_image3D(img_dat['scn_to_wrld_tx'], XY_mat_mm)
    resampled_Z_mm = resampler.resample(img_dat['x'], img_dat['y'], img_dat['z'])
    img_dat.update({'stl_z': true_Z_mat_mm, 'height_mat_mm': resampled_Z_mm})
    
def add_true_z_values_and_save(images_4D_dat_path, stl_path, out_path, max_z_mm):
    
    sampler = Image3Dsampler(stl_path, max_z_mm)
    images_4D_dat = np.load(images_4D_dat_path).tolist()
    fill_all_XYT(images_4D_dat)
    add_true_z_heights(images_4D_dat, sampler)
    np.save(out_path, images_4D_dat)

def add_true_z_heights(images_4D_dat, sampler):
    
    for img_dat in images_4D_dat.itervalues():
        add_true_z_to_img(img_dat, sampler)
        
def add_true_z_to_img(img_dat, sampler):
    
    if 'scn_to_wrld_tx' not in img_dat:
        print(str(img_dat['scan_hardware_id']) + ' has no scn_to_wrld_tx')
        return
    XY_mat_mm = np.array((img_dat['x'], img_dat['y']))
    true_Z_mat_mm = sampler.get_image3D(img_dat['scn_to_wrld_tx'], XY_mat_mm)
    img_dat['stl_z'] = true_Z_mat_mm
    
def fill_all_XYT(imgs_dat):
    from .XYT_filling import XYTFiller
    
    filler = XYTFiller(SHAPE)
    for img_d in imgs_dat.itervalues():
        fill_particular_XYT(img_d, filler)
        
def fill_particular_XYT(img_d, filler):
    
    if 'scn_to_wrld_tx' not in img_d:
        return
    new_XYT = filler.fill_XYT(img_d['x'], img_d['y'], img_d['t'], img_d['z'])
    img_d.update(new_XYT)