import os
import h5py
from pprint import pprint
from datetime import datetime
import numpy as np
from typing import Dict, Iterable

def build_hdf5(name="default.h5", groups=[], path="."):
    hdf5_path = os.path.join(path, name)  

    with h5py.File(hdf5_path, "a") as f:
        # create an empty h5py folder with a group for each probe
        for gr in groups:
            f.create_group(gr)
        

def get_all_h5():
    return [file for file in os.listdir(".") if file[-3:]==".h5"]

def see_groups_and_datasets(filepath, subgroup=None):
    with h5py.File(filepath, "r") as f:
        if subgroup:
            f = f[subgroup]
        group_keys = [key for key, items in f.items() if isinstance(items, h5py.Group)]
        dataset_keys = [key for key, items in f.items() if isinstance(items, h5py.Dataset)]
    return dict(group_keys=group_keys, dataset_keys=dataset_keys)

def add_group(hdf5_path, group, **kargs):
    assert not group_exist(hdf5_path, group), "group already exists"
    # add the group to the library
    with h5py.File(hdf5_path, "a") as f:
        group = f.create_group(group)
        group.attrs.update(**kargs)
    
    print(see_groups_and_datasets(hdf5_path)["group_keys"])

def remove_group(hdf5_path, group):
    assert group_exist(hdf5_path, group), "group does not exist"
    with h5py.File(hdf5_path, "a") as f:
        del f[group] 


    
def group_exist(hdf5_path, group):
    assert(exists(hdf5_path))
    group_keys = see_groups_and_datasets(hdf5_path)["group_keys"]
    return (group in group_keys)
    
def exists(hdf5_path):
    return (hdf5_path in get_all_h5())

def printall(name, obj):
        print("NAME: {:^30}".format(name))
        print("Type: {:^20}".format(f"GROUP - Subgroups: {list(obj.keys())}" if isinstance(obj, h5py.Group) else "DATASET"))
        print("Parent Path: {:<10}".format(obj.parent.name))
        print("Attributes: ")
        pprint(dict(obj.attrs))
        if isinstance(obj, h5py.Dataset):
            print("shape: ", obj.shape, "____ dtype: ", obj.dtype) 
        print("\n\n\n") 

def explore_library(path, recursive=True):
    
    with h5py.File(path, "r") as f:
        if recursive:
            f.visititems(printall)
        else:
            for name, obj in f.items():
                printall(name, obj)




def save_field_to_hdf5(group, name, field, **properties):
    now = datetime.now()
    dt_string = now.strftime("%d__%m__%Y %H:%M:%S")

    
    # create field dataset
    field_ds = group.create_dataset(name=name, shape = field.shape, data = field)
    field_ds.attrs.update(properties)
    field_ds.attrs.update({"creation date": dt_string})


def save2hdf5(full_h5path: str, complex_fields: Dict[str, np.ndarray], frequencies: np.ndarray, grid: Iterable, savename: str = "", **kargs):
    now = datetime.now()
    dt_string = now.strftime("%d__%m__%Y %H:%M:%S")

    if savename == "":
        savename=dt_string

    if not os.path.exists(full_h5path):
        build_hdf5(name=full_h5path)

    field_names = list(complex_fields.keys())
    
    


    with h5py.File(full_h5path, "a") as f:
        measurement_group_keys = [key for key, items in f.items() if isinstance(items, h5py.Group)]

        if not savename in measurement_group_keys:
            big_group = f.create_group(savename)
            big_group.attrs.update({"creation_date": dt_string})
        

        f = f[savename]
        for gr in field_names:
            
            # automatically overwrite existing groups
            group_keys = [key for key, items in f.items() if isinstance(items, h5py.Group)]
            if gr in group_keys:
                del f[gr]

            g = f.create_group(gr) 
            g.attrs["creation date"]= dt_string
          
            f_ds = g.create_dataset(name="freqs", data = np.array(frequencies), shape=np.array(frequencies).shape, **kargs)
            f_ds.attrs.update({"unit":"Hz"})

            # create coordinate dataset
            grid_ds = g.require_dataset(name="grid", shape = np.array(grid).shape, dtype = np.float32, data = np.array(grid), **kargs)  
            grid_ds.attrs["SI unit"] = "m" 

            # create field dataset
            field_ds = g.require_dataset(name="field", shape = complex_fields[gr].shape, dtype=np.complex128, data = complex_fields[gr], **kargs)
            field_ds.attrs.update({"grid": "check grid dataset", "frequencies": "check frequency dataset"})