
import glob
import os
from true_data_image_4D_resampler import FILES_NAMES, resample_height_map_and_add_true_z

SHAPE = (155, 208)
STL_NAME = '{}_jaw.stl'

def create_all_image_data(root_folder):
    
    folders = glob.glob(os.path.join(root_folder, '*', '*'))
    for fold in folders:
        fnames = get_file_names(fold)
        resample_height_map_and_add_true_z(fold, fnames = fnames)

def get_file_names(fold):
    
    case_type = fold.replace('\\', '/').split('/')[-1]
    fnames = FILES_NAMES.copy()
    if 'lower' in case_type:
        fnames['stl'] = STL_NAME.format('lower')
    elif 'upper' in case_type:
        fnames['stl'] = STL_NAME.format('upper')
    else:
        raise NameError('Folder should contain valid case type')
    return fnames