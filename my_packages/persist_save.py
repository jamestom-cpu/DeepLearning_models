import json
from types import SimpleNamespace

import numpy as np

# save all information to a json file

class Model2JSON:
    def __init__(
            self, 
            dfh,
            **kwargs
            ):
        
        self.substrate = {}
        self.EM_space = {}
        self.dipoles = {}

        self.substrate.update(dfh.substrate.__dict__)
        self.EM_space.update(dfh.EM_space.__dict__)
        self.EM_space["r"] = np.asarray(self.EM_space["r"])
        self.f = dfh.dipole_array.f
        
        dipole_types = [str(d.type) for d in dfh.dipole_array.dipoles]
        r0 = np.asarray(dfh.dipole_array.r0)
        M = np.abs(dfh.dipole_array.M)
        phases = np.angle(dfh.dipole_array.M)
        orientations = dfh.dipole_array.orientations

        self.dipoles.update({"dipole_types": dipole_types, "r0": r0, "M": M, "phases": phases, "orientations": orientations})

        # finally update with extra kwargs overwriting the above if necessary
        self.__dict__.update(kwargs)

    def to_json(self, filepath):
        with open(filepath, "w") as f:
            json.dump(convert_dict_to_numpy(self.__dict__), f, sort_keys=True, indent=4)
        print("Saved to {}".format(filepath))


def convert_dict_to_numpy(d):
    # Create an empty dictionary to store the output
    out_dict = {}

    # Iterate over the input dictionary
    for k, v in d.items():
        # If the value is a dictionary, recursively call the function on the value
        if isinstance(v, dict):
            out_dict[k] = convert_dict_to_numpy(v)
        # If the value is a numpy array, add it to the output dictionary
        elif isinstance(v, np.ndarray):
            out_dict[k] = v.tolist()
        # If the value is a list or tuple, convert its elements to numpy arrays recursively
        elif isinstance(v, (list, tuple)):
            out_dict[k] = v
        # For all other types of values, add them to the output dictionary as is
        else:
            out_dict[k] = v

    return out_dict



