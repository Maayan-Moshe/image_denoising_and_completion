
from pycaster import pycaster
import numpy as np

class Image3Dsampler:
    
    def __init__(self, stl_path, max_z_mm, extra_rotation = np.eye(3)):
        '''
        We assume that max_z_mm is the inside the wand.
        '''
        self.caster = pycaster.rayCaster.fromSTL(stl_path, scale=1)
        self.max_z_mm = max_z_mm
        self.extra_rotation = extra_rotation
        
    def get_image3D(self, scan_to_world_tx, XY_mat_mm):
        '''
        expect tha transformation XY_mat (2, num_rows, num_cols), max_z_mm should be about 25mm
        '''
        scn_to_wrld = np.dot(self.extra_rotation,scan_to_world_tx)
        world_to_scan_tx = np.linalg.inv(scn_to_wrld)
        Z_mat_mm = np.zeros((XY_mat_mm.shape[1], XY_mat_mm.shape[2]))
        for row in range(XY_mat_mm.shape[1]):
            for col in range(XY_mat_mm.shape[2]):
                Z_mat_mm[row, col] = self.__get_z_value(XY_mat_mm[:, row, col],
                                    scn_to_wrld, world_to_scan_tx)
        return Z_mat_mm
        
    def __get_z_value(self, xy_mm, scan_to_world_tx, world_to_scan_tx):
        
        extreme_pnt_scan = np.array(((xy_mm[0], xy_mm[1], 0), (xy_mm[0], xy_mm[1], self.max_z_mm)))
        extreme_pnt_world = np.dot(extreme_pnt_scan, scan_to_world_tx[:3,:3].T) + scan_to_world_tx[:3,3]
        intrsctn_pnts_wrld = self.caster.castRay(extreme_pnt_world[0], extreme_pnt_world[1])
        if len(intrsctn_pnts_wrld) == 0:
            return 0
        pnt_scn = np.dot(world_to_scan_tx[:3,:3], intrsctn_pnts_wrld[-1]) + world_to_scan_tx[:3,3]
        error = np.linalg.norm(pnt_scn[:2] - xy_mm)
        assert error < 1e-5, 'xy values should not be changed error - ' + str(error)
        return pnt_scn[2]