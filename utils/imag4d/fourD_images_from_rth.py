import glob

import numpy as np
import os

import research.sayal.imag4d.image4d_deserializer
import pyrth.tests.common.rthdriver as rthdriver

# From CarriesDetection
import util.rthdeserializer
from util.TI_images_from_rth import get_scans_pos_as_dict


def prepare_4d_unified(model_folder):
    rth_fname = glob.glob(os.path.join(model_folder, '*.rth'))[0]
    scans = get_image4d_and_tx(rth_fname, model_folder)
    np.save(os.path.join(model_folder, "4d_images.npy"), scans)


def prepare_4d(record_name, model_toplevel_folder=r"C:\iTero\models", data_toplevel_folder=r"\\fs05\Shared Box\Image4D_database"):
    data_folder = os.path.join(data_toplevel_folder, record_name)
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    model_folder = os.path.join(model_toplevel_folder, record_name)
    model = os.path.join(model_folder, "{}.rth".format(record_name))
    scans = get_image4d_and_tx(model, model_folder)
    np.save(os.path.join(data_folder, "4d_images.npy"), scans)


def save_stl_from_model(model, outfile):
    setting = rthdriver.create_settings_for_recfolder_playback({
            'cmdline_arguments.model_to_open': model,
            "scan_handling.keep_intermediate_scan_data": True
         })
    
    rth = rthdriver.RthDriver()
    rth.run(settings=setting, build_to_use=r'D:\git2rep\iTeroRTH\bin\Release\x64')
    rth.ui_open_view_screen()
    rth.export_merge_as_surface(rthdriver.LOWER_JAW, outfile)
    rth.kill()


def get_image4d_and_tx(rth_fname, tx_folder):
    positions = _get_4d_positions_from_rth(rth_fname, tx_folder)
    scans = research.sayal.imag4d.image4d_deserializer.deserialize_rectificated_scans(rth_fname)
    for k in positions:
        if k in scans:
            scans[k]['scn_to_wrld_tx'] = positions[k]['position tx']

    return scans


def _get_4d_positions_from_rth(rth_file, tx_folder):
    scan_id_to_tx = research.sayal.imag4d.image4d_deserializer.deserialize_blend_txs(rth_file, tx_folder)
    cps = util.rthdeserializer.deserialize_canonical_positioning_state(rth_file)
    scans_pos = get_scans_pos_as_dict(cps)
    positions = _get_4d_positions_to_world(scan_id_to_tx, scans_pos)
    return positions


def _get_4d_positions_to_world(scan_id_to_tx, scans_pos):
    positions = dict()
    for scan_id in scan_id_to_tx:
        blend_scan_id = scan_id_to_tx[scan_id]['blend_scan_id']
        scn_pos = scans_pos.get(blend_scan_id, np.empty(0))
        if len(scn_pos) > 0:
            tx = np.dot(scans_pos[blend_scan_id], scan_id_to_tx[scan_id]['relative_pos'])
            positions[scan_id] = {'position tx': tx}
    return positions



RTH_FILES_FOLDER = r"\\fs05\Shared Box\Image4D_database\IO"

if __name__ == "__main__":
    for folder in glob.glob(os.path.join(RTH_FILES_FOLDER, '*', '*')):
        print(folder)
        if not os.path.isfile(os.path.join(folder, "4d_images.npy")):
            prepare_4d_unified(folder)
