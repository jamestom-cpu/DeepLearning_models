import os
from types import SimpleNamespace
import numpy as np

from my_packages.hdf5.util_classes import Measurement_Handler




# load scanned data
class FieldLoader():
    def __init__(self, db_directory, db_filename, db_group_name, probe_height=6e-3):
        self.db = SimpleNamespace(directory=db_directory, filename=db_filename, savename=db_group_name)
        self.probe_height = probe_height
        self.run_scans_on_database_field()
    
    def run_scans_on_database_field(self):
         # hdf5 database properties
        fullpath = os.path.join(self.db.directory, self.db.filename)

        # load the database properties to mhandler
        self.m_handler = Measurement_Handler.from_h5_file(fullpath, self.db.savename)

        # create the target fields: Ez, Hx, Hy only magnitudes on a plane
        self.Ez = self.m_handler.E.run_scan("z", field_type="E", index = self.probe_height)
        self.Hx = self.m_handler.H.run_scan("x", field_type="H", index = self.probe_height)
        self.Hy = self.m_handler.H.run_scan("y", field_type="H", index = self.probe_height)

        assert self.Ez.f == self.Hx.f == self.Hy.f, "All fields must have the same frequency"
        assert np.allclose(self.Ez.grid, self.Hx.grid) and np.allclose(self.Ez.grid, self.Hy.grid), "All fields must have the same grid"
        self.f = [self.Ez.f] # assuming all fields have the same frequency
        self.scan_grid = np.expand_dims(self.Ez.grid, axis=-1)  # the scan grid is 2D, we need to add the third dimension
