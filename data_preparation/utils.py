
def get_all_npy_from_folder(folder):
    import os
    
    fnames = list()
    for fname in os.listdir(folder):
        if fname.endswith(".npy"):
            fnames.append(fname)
    
    return fnames