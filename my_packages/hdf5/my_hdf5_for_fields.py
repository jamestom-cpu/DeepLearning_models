
from typing import List, Dict
from dataclasses import dataclass, field
import os
import numpy as np
import h5py

from . import my_hdf5
import pyaedt
from my_packages.classes.field_classes import Field3D

@dataclass()
class Measurement_Handler():
    E: Field3D = None
    H: Field3D = None
    measurement_date: str = "Not Set"
    attributes: dict = field(default_factory={})

    @staticmethod
    def from_h5_file(h5path: str, measurement_name: str):
        return get_fields_from_h5(h5path, measurement_name)


def save_fields_to_h5(fields: List[Field3D], full_h5path: str, savename:str, hfss: pyaedt.Hfss =None, overwrite=False, attrs={}):
    
    field_dict = {field.__name__: field.field for field in fields}


    if os.path.exists(full_h5path):
        first_level_content = my_hdf5.see_groups_and_datasets(full_h5path)

        assert all(["__name__" in dir(field) for field in fields]), AttributeError("All fields must have a `__name__` attribute")


        if savename in first_level_content["group_keys"]:
            if overwrite:
                print("overwriting ...")
            else:
                print("savename already present")
                return 0
                

    
    my_hdf5.save2hdf5(
            full_h5path, complex_fields=field_dict, frequencies=fields[0].freqs, grid = fields[0].grid, 
            savename=savename, compression=6, chunks=True
            )

    if hfss is not None:
        with h5py.File(full_h5path, "a") as h5file:
            group = h5file[savename]
            group.attrs.update(hfss.available_variations.nominal_w_values_dict)
            group.attrs.update(attrs)
    return 1

def get_fields_from_h5(full_h5path: str, measurement_name: str = None, show_print=False) -> Measurement_Handler:
    """
    this func is built to handle measurements where both E and H are present and under the names:
        * 'E'
        * 'H'
    """
    with h5py.File(full_h5path, "r") as f:
        
        if measurement_name is None: 
            groups = my_hdf5.see_groups_and_datasets(full_h5path)["group_keys"]
            print("the file contains the following groups:")
            print(groups)

            print(f"selecting group {groups[0]}")
            measurement_name = groups[0]

        db = f[measurement_name] 
        if show_print:
            db.visititems(my_hdf5.printall)   

        E = Field3D(np.array(db["E/field"]), freqs = np.array(db["E/freqs"]), grid = np.array(db["E/grid"]))
        H = Field3D(np.array(db["H/field"]), freqs = np.array(db["E/freqs"]), grid = np.array(db["H/grid"]))
        attributes = dict(db.attrs)
        measurement_date = attributes.pop("creation_date", None)
        return Measurement_Handler(E=E, H=H, measurement_date=measurement_date, attributes=attributes)
        

        

     


