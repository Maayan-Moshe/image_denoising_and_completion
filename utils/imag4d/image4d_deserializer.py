import glob
import os
import struct
import numpy as np
import cStringIO as StringIO

from pyrth.utils.rth_file_utils import RTHdb, does_key_exist


def deserialize_single_blend_tx(buf, scan_id_to_tx):
    scan_to_tx = struct.Struct('IIII16f')
    des = scan_to_tx.unpack(buf)
    scan_id_to_tx[des[1]] = {
        'scan_hardware_id': des[2],
        'blend_scan_id': des[3],
        'relative_pos': np.reshape(des[4:], (4, 4)),
    }
    return scan_id_to_tx


def deserialize_image4D(f):
    header = struct.Struct('II')
    footer = struct.Struct('IffffB')
    image4D = {}

    rows, cols = header.unpack(f.read(header.size))
    image4D['x'] = np.frombuffer(f.read(rows * cols * 4), dtype='float32', count=rows * cols).reshape(rows, cols)
    rows, cols = header.unpack(f.read(header.size))
    image4D['y'] = np.frombuffer(f.read(rows * cols * 4), dtype='float32', count=rows * cols).reshape(rows, cols)
    rows, cols = header.unpack(f.read(header.size))
    image4D['z'] = np.frombuffer(f.read(rows * cols * 4), dtype='float32', count=rows * cols).reshape(rows, cols)
    rows, cols = header.unpack(f.read(header.size))
    image4D['t'] = np.frombuffer(f.read(rows * cols * 4), dtype='float32', count=rows * cols).reshape(rows, cols)

    im_footer = dict(zip(
        ['scan_hardware_id', 'lowres_time_stamp_msec', 'scan_duration_msec', 'effective_scan_duration_msec', 'lateral_resolution_mm', 'lens_dir'],
        footer.unpack(f.read(footer.size))
    ))
    image4D.update(im_footer)
    return image4D


def deserialize_blend_tx_from_folder(folder):
    scan_id_to_tx = {}

    for blend_file in glob.glob(os.path.join(folder, "blend_tx*.bin")):
        with open(blend_file, 'rb') as f:
            scan_id_to_tx = deserialize_single_blend_tx(f.read(), scan_id_to_tx)

    return scan_id_to_tx


def deserialize_blend_tx_from_rth(rth_fname):
    scan_id_to_tx = {}
    db = RTHdb(rth_fname)
    blend_txs = db.fetchall_by_key_base('blend_tx')

    for blend in blend_txs:
        scan_id_to_tx = deserialize_single_blend_tx(blend[3], scan_id_to_tx)

    return scan_id_to_tx


def deserialize_rectificated_scans(rth_fname):
    db = RTHdb(rth_fname)
    scans = db.fetchall_by_key_base('rectificated_scan')
    header = struct.Struct('I')
    rectified_scans = {}
    for l in scans:
        f = StringIO.StringIO(l[3])
        scan_id,  = header.unpack(f.read(header.size))
        image4D = deserialize_image4D(f)
        rectified_scans[scan_id] = image4D

    return rectified_scans


def deserialize_blend_txs(rth_fname, tx_folder):
    if does_key_exist('blend_tx'):
        return deserialize_blend_tx_from_rth(rth_fname)
    else:
        return deserialize_blend_tx_from_folder(tx_folder)



