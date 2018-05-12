
import numpy as np
from scipy.interpolate import griddata

class XYTFiller:
    
    def __init__(self, shape):
        
        self.i, self.j = np.indices(shape)
        
    def fill_XYT(self, X, Y, T, Z):
        
        filler = self.__get_grid_filler(Z)
        new_X = filler.get_filled(X)
        new_Y = filler.get_filled(Y)
        new_T = filler.get_filled(T)
        return {'x': new_X, 'y': new_Y, 't': new_T}
        
    def __get_grid_filler(self, Z):
        
        valid = get_valid_mask(Z)
        not_v = np.logical_not(valid)
        points = np.array((self.i[valid].ravel(), self.j[valid].ravel())).T
        sample_pnts = np.array((self.i[not_v].ravel(), self.j[not_v].ravel())).T
        filler = GridFiller(points, valid, sample_pnts)
        return filler
                    
class GridFiller:
    
    def __init__(self, points, valid, sample_pnts):
        
        self.points = points
        self.valid = valid
        self.sample_pnts = sample_pnts
        
    def get_filled(self, mat):

        fill_X = griddata(self.points, mat[self.valid].ravel(), self.sample_pnts)
        new_mat = np.copy(mat)
        new_mat[self.sample_pnts[:,0], self.sample_pnts[:,1]] = fill_X
        return new_mat
        
def get_valid_mask(Z):
    
    valid = Z > 0
    valid[0,  :] = True
    valid[-1, :] = True
    valid[:,  0] = True
    valid[:, -1] = True
    return valid
        