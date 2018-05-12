import numpy as np

'''This is the ordering assumed in the feature channel''' 
AX_INDEXES = {'t': 0, 'x': 1, 'y': 2, 'z': 3}

def prepare_image_data(fpath):
    
    orig_data = np.load(fpath).tolist()
    images = list()
    true_z = list()
    for val in orig_data.itervalues():
        append_image(val, images, true_z)
    return {'images': np.array(images), 'stl_z': np.array(true_z)}
    
def append_image(val, images, true_z):
    '''
    It is assumed but not enforced that the ordering is the same as in AX_INDEXES.
    '''
    if 'stl_z' not in val:
        return
    images.append(val['height_mat_mm'])
    true_z.append(val['stl_z'])

def prepare_scan_data(fpath):
    
    orig_data = np.load(fpath).tolist()
    txyz = list()
    true_z = list()
    for val in orig_data.itervalues():
        append_z_map(val, txyz, true_z)
    return {'image_4d_xyzt': np.array(txyz), 'stl_z': np.array(true_z)}

def append_z_map(val, txyz, true_z):
    '''
    It is assumed but not enforced that the ordering is the same as in AX_INDEXES.
    '''
    if 'stl_z' not in val:
        return
    new_txyz = np.array([val['t'], val['x'], val['y'], val['z']])
    new_txyz = np.moveaxis(new_txyz, [0, 1, 2], [2, 0, 1])
    txyz.append(new_txyz)
    true_z.append(val['stl_z'])
    
def get_z_data_for_learning(fpath):
    
    all_data = prepare_scan_data(fpath)
    z = all_data['image_4d_xyzt'][:, :, :, AX_INDEXES['z']]
    true_z = all_data['stl_z']
    return z, true_z