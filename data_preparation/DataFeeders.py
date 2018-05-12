from .data_preparation import get_z_data_for_learning
from .utils import get_all_npy_from_folder
import numpy as np
import os

class OneFileValidatorFeeder:
    
    def __init__(self, data_path, cropping, **kwargs):
        
        data = np.load(data_path, encoding = 'latin1').tolist()
        self.__set_data(data)
        self.cropping = cropping

    def get_all_data(self):
        
        return {'input': self.image[:, :, self.cropping:], 'truth': self.true_z[:, :, self.cropping:]}
        
    def __set_data(self, data):

        assert data['images'].shape == data['stl_z'].shape, \
                    'Dimensions of input and truth do not match'
        self.image = data['images']
        self.true_z = data['stl_z']

class OneFileZDataFeeder:
    
    def __init__(self, data_path, batch_size, **kwargs):
        
        print('unused parameters: ', kwargs.keys())
        z, true_z = get_z_data_for_learning(data_path)
        self.data = np.array((z, true_z))
        self.batch_size = batch_size
        self.count = 0
    
    def next_batch(self):
        
        batch = np.roll(self.data, self.batch_size*self.count, axis = 1)[:, :self.batch_size, :, :]
        self.count += 1
        return {'input': batch[0], 'truth': batch[1]}
        
class OneFileTXYZDATAFeeder:
    
    def __init__(self, data_path, batch_size, cropping, **kwargs):
        
        data = np.load(data_path, encoding = 'latin1').tolist()
        self.__set_shuffled_data(data)
        self.batch_size = batch_size
        self.tot_size = 0
        self.cropping = cropping
        
    def get_all_data(self):
        
        return {'input': self.xyzt[:, :, self.cropping:, :], 'truth': self.true_z[:, :, self.cropping:]}
        
    def next_batch(self):
    
        xyzt_batch = np.roll(self.xyzt, self.tot_size, axis = 0)[:self.batch_size, :, self.cropping:, :]
        true_z_batch = np.roll(self.true_z, self.tot_size, axis = 0)[:self.batch_size, :, self.cropping:]
        self.tot_size += self.batch_size%self.tot_length
        return {'input': xyzt_batch, 'truth': true_z_batch}
    
    def __set_shuffled_data(self, data):
        
        assert data['image_4d_xyzt'].shape[:-1] == data['stl_z'].shape, \
                    'Dimensions of input and truth do not match'
        self.tot_length = data['stl_z'].shape[0]
        order = np.random.permutation(range(self.tot_length))
        self.xyzt = data['image_4d_xyzt'][order]
        self.true_z = data['stl_z'][order]
        
class FolderTXYZDataFeeder:
    
    def __init__(self, folder, batch_size, **kwargs):
        
        self.folder = folder
        self.__set_all_files_in_folder()
        self.image_index = 0
        self.batch_size = batch_size
        
    def next_batch(self):
        
        start, end = self.__get_start_end()
        xyzt_batch = self.xyzt[start:end]
        true_z_batch = self.true_z[start:end]
        if end == self.file_length:
            xyzt_batch, true_z_batch = self.__next_file(xyzt_batch, true_z_batch)
        return  {'input': xyzt_batch, 'truth': true_z_batch}
    
    def __get_start_end(self):
        
        start = self.image_index
        self.image_index += self.batch_size
        end = min(self.image_index, self.file_length)
        return start, end
    
    def __set_all_files_in_folder(self):
                
        self.fnames = get_all_npy_from_folder(self.folder)
        self.file_index = 0
        self.__set_data_from_file()
        
    def __next_file(self, xyzt_batch, true_z_batch):
        
        self.__set_data_from_file()
        length = self.batch_size - len(xyzt_batch)
        xyzt_batch = np.vstack((xyzt_batch, self.xyzt[:length]))
        true_z_batch = np.vstack((true_z_batch, self.true_z[:length]))
        self.image_index = length
        return xyzt_batch, true_z_batch
    
    def __set_data_from_file(self):
        
        data_path = os.path.join(self.folder, self.fnames[self.file_index])
        data = np.load(data_path, encoding = 'latin1').tolist()
        self.__set_shuffled_data(data)
        self.file_index += 1
        self.file_index %= len(self.fnames)
        
    def __set_shuffled_data(self, data):
        
        assert data['image_4d_xyzt'].shape[:-1] == data['stl_z'].shape, \
                    'Dimensions of input and truth do not match'
        self.file_length = data['stl_z'].shape[0]
        order = np.random.permutation(range(self.file_length))
        self.xyzt = data['image_4d_xyzt'][order]
        self.true_z = data['stl_z'][order]
        
class SeparateFilesFeeder:
    
    def __init__(self, folder, batch_size, cropping, **kwargs):
        
        self.folder = folder
        self.__set_files_list_in_folder()
        self.image_index = 0
        self.batch_size = batch_size
        self.cropping = cropping
        
    def next_batch(self):
        
        xyzt_batch = list()
        true_z_batch = list()
        for index in range(self.batch_size):
            add_txyz, add_stlz = self.__read_file()
            xyzt_batch.append(add_txyz)
            true_z_batch.append(add_stlz)
        return {'input': xyzt_batch, 'truth': true_z_batch}
        
    def __read_file(self):
        
        fname = self.fnames[self.order[self.image_index]]
        fpath = os.path.join(self.folder, fname)
        fdata = np.load(fpath, encoding = 'latin1').tolist()
        self.__advance_index()
        return fdata['image'][:, self.cropping:], fdata['stl_z'][:, self.cropping:]
        
    def __advance_index(self):
        
        self.image_index += 1
        self.image_index %= self.num_files
               
    def __set_files_list_in_folder(self):
        
        self.fnames = get_all_npy_from_folder(self.folder)
        self.num_files = len(self.fnames)
        self.order = np.random.permutation(range(self.num_files))

class MINSTSingleFileRandomZeros:

    def __init__(self, file_path, batch_size, zero_percentage, **kwargs):
        from .reading_minst_data import read_idx

        self.true_data = read_idx(file_path)
        self.order = np.random.permutation(len(self.true_data))
        self.batch_size = batch_size
        self.index = 0
        self.zero_percentage = zero_percentage
        assert 1 >= self.zero_percentage >= 0, 'percentage should be between 0 and 1'

    def next_batch(self):
        true_batch = self.__get_true_data()
        corrupted_batch = self.__get_corrupted_data(true_batch)
        return {'input': corrupted_batch, 'truth': true_batch}

    def __get_true_data(self):
        true_batch = self.true_data[self.order[self.index: self.index + self.batch_size]]
        self.index += self.batch_size
        self.index %= len(self.true_data)
        return true_batch

    def __get_corrupted_data(self, true_batch):
        corrupted_batch = np.copy(true_batch)
        sze = corrupted_batch.size
        zero_indexes = np.random.choice(range(sze), int(self.zero_percentage * sze))
        corrupted_batch.ravel()[zero_indexes] = 0
        return corrupted_batch
