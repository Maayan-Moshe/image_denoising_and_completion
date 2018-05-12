
import numpy as np

class Image4DToHeightMapResampler:
    
    def __init__(self, xy_resolution_mm = 0.09, shape = (178, 233)):
        
        self.xy_resolution = xy_resolution_mm
        self.shape = shape
        self.__set_xy_matrices()
        
    def resample(self, X, Y, Z):

        pos_x, pos_y, pos_z = get_positive_xyz(X, Y, Z)
        height_map, x_indexes, y_indexes = self.__get_cumulative_z_values(pos_x, pos_y, pos_z)
        return height_map
        
    def get_xy_matrices(self):
        
        return np.array((self.X, self.Y))
        
    def __set_xy_matrices(self):
        
        x = np.array(range(self.shape[0]))*self.xy_resolution
        y = np.array(range(self.shape[1]))*self.xy_resolution
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
    def __get_cumulative_z_values(self, pos_x, pos_y, pos_z):
        
        height_map = np.zeros(self.shape)
        x_indexes, T = get_indexes_from_positions(pos_x, self.xy_resolution, self.shape[0])
        y_indexes, S = get_indexes_from_positions(pos_y, self.xy_resolution, self.shape[1])
        Q = np.logical_and(T, S)
        height_map[x_indexes[Q], y_indexes[Q]] += pos_z[Q]
        return height_map, x_indexes[Q], y_indexes[Q]
        
def get_indexes_from_positions(pos, res, max_len):
    
    indexes = np.round(pos/res).astype(int)
    T = indexes < max_len
    return indexes, T
        
def get_positive_xyz(X, Y, Z):
    
    T = Z > 0
    pos_x = X[T].ravel()
    pos_y = Y[T].ravel()
    pos_z = Z[T].ravel()
    return pos_x, pos_y, pos_z