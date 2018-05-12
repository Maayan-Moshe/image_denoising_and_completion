import glob
import os
import shutil
import zipfile
import tempfile
import re

import pyrth.tests.common.rthdriver as rthdriver
from pyrth.rthdbg.classes import scan_role


def copy_rth_files(model_folder, dest):
    rth_files = glob.glob(os.path.join(model_folder, '*.*'))
    for f in rth_files:
        shutil.copy(f, dest)


def unzip_to_tmp(input_zip_file):
    tmp_folder = tempfile.mktemp(dir='c:/temp')
    zip_ref = zipfile.ZipFile(input_zip_file, 'r')
    zip_ref.extractall(tmp_folder)
    zip_ref.close()
    return tmp_folder


def run_rth_on_recording(record_folder, jaw, build):
    setting = rthdriver.create_settings_for_recfolder_playback({"scan_handling.keep_intermediate_scan_data": True})
    rth = rthdriver.RthDriver()
    rth.run(settings=setting, build_to_use=build)
    rth.ui_set_scanning_role(scan_role.to_int(jaw))
    rth.play_hw_recfolder(record_folder)

    rth.reset_event(rthdriver.POST_PROCESSING_DONE_EVENT)
    rth.ui_open_view_screen()
    rth.wait_for_notification(rthdriver.POST_PROCESSING_DONE_EVENT)

    model_folder = rth.get_current_model_folder()
    stl_fname = os.path.join(model_folder, jaw + '.stl')

    rth.export_merge_as_surface(scan_role.to_int(jaw), stl_fname)
    
    rth.ui_open_send_screen()

    #rth.kill()
    return model_folder


def get_name_and_jaw_from_filename(record_name):

    path_split = os.path.normpath(record_name).split(os.sep)
    name = path_split[-3]
    jaw = re.match(r".*_(upper|lower)_jaw.*", record_name).groups()[0] + '_jaw'
    return name, jaw


RECORD_ROOT_FOLDER = r"\\prilstg01\QA\new_Accuracy_dataset_packed\IO"
MODEL_ROOT_FOLDER = r'C:/Users/mmoshe/Documents/teeth_segmentation/IO_example'#r"\\fs05\Shared Box\Image4D_database\IO"


def process_recording(recording, build):
    name, jaw = get_name_and_jaw_from_filename(recording)
    output_path = os.path.join(MODEL_ROOT_FOLDER, name, jaw)
    if not os.path.isdir(output_path):

        tmp_record_folder = unzip_to_tmp(recording)
        model_folder = run_rth_on_recording(tmp_record_folder, jaw, build)

        shutil.copytree(model_folder, output_path)
        shutil.rmtree(tmp_record_folder)
        shutil.rmtree(model_folder)


if __name__ == "__main__":
    recs = glob.glob(os.path.join(RECORD_ROOT_FOLDER, '*', '*', 'record', '*.zip'))
    for rec in recs[:1]:
        print(rec)
        process_recording(rec, build='installed')
        
