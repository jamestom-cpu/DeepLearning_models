import numpy as np
import os 
import datetime
import h5py

from my_packages.hdf5 import my_hdf5

def get_fields_from_h5(full_h5path, group, show_print=False, get_freqs = False):
    with h5py.File(full_h5path, "r")as f:
        
        db = f[group] 

        if show_print:
            db.visititems(my_hdf5.printall)

        E = np.array(db["E field/E"])
        H = np.array(db["H field/H"])
        grid = np.array(db["E field/grid"])

        attr = dict(db["E field"].attrs)
        solved_freqs = attr["frequencies"]

        if get_freqs:
            return E, H, grid, np.array(solved_freqs)

    return E, H, grid


def save2hdf5(full_h5path, complex_fields, frequencies, grid, savename=None, **kargs):
    now = datetime.now()
    dt_string = now.strftime("%d__%m__%Y %H:%M:%S")

    if savename is None:
        savename=dt_string

    if not os.path.exists(full_h5path):
        my_hdf5.build_hdf5(name=full_h5path)

    groups = list(complex_fields.keys())
        


    with h5py.File(full_h5path, "a") as f:

        if not savename in groups:
            big_group = f.create_group(savename)
            big_group.attrs.update({"creation_date": dt_string})

        f = f[savename]
        for gr in groups:
            g = f.create_group(f"{gr} field") 
            g.attrs["creation date"]= dt_string
            g.attrs["frequencies"] = frequencies
            
            freqs = np.array(frequencies)
            f_ds = g.create_dataset(name="solution_frequencies", data = freqs, shape=freqs.shape, **kargs)
            f_ds.attrs.update({"unit":"Hz"})

            # create coordinate dataset
            grid_ds = g.require_dataset(name="grid", shape = np.array(grid).shape, dtype = np.float32, data = np.array(grid), **kargs)  
            grid_ds.attrs["SI unit"] = "mm" 

            # create field dataset
            field_ds = g.require_dataset(name=gr, shape = complex_fields[gr].shape, dtype=np.complex128, data = complex_fields[gr], **kargs)
            field_ds.attrs.update({"grid": "check grid dataset", "frequencies": "check frequency dataset"})