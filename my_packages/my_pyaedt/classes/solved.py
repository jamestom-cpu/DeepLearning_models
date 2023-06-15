import os
import numpy as np
import copy
import shutil
import pyaedt
from typing import Iterable
from pprint import pprint

from my_packages.my_pyaedt import utils
from my_packages.hdf5.my_hdf5_for_fields import save_fields_to_h5
from my_packages.classes import field_classes

from dataclasses import dataclass

@dataclass
class CoordinateSystem():
    name: str
    z_shift: float

def check_coordinate_systems(hfss):
        BaseCS = [CoordinateSystem("Global", 0)]
        OtherCS = [
            CoordinateSystem(
                CS.name,
                CS.props["OriginZ"]
                ) for CS in hfss.modeler.coordinate_systems]
        allCS = BaseCS+OtherCS
        return allCS



class SolvedDesign():
    def __init__(self, hfss:pyaedt.Hfss, setup:str = "", sweep:str = "", coordinate_system="Global"):
        self.hfss = hfss
        self.setup = hfss.setup_names[0] if setup == "" else setup
        self.sweep = hfss.get_sweeps(self.setup)[0] if sweep=="" else sweep
        self._check_coordinate_systems()
        self.working_coordinate_system = coordinate_system
        
    
    @property
    def working_coordinate_system(self):
        return self._working_coordinate_system
    
    @working_coordinate_system.setter
    def working_coordinate_system(self, CS_name):
        self.set_coordinate_system(CS_name)

        
    
    @property
    def coordinate_systems(self):
        return self._coordinate_systems

    @coordinate_systems.setter
    def coordinate_systems(self, _):
        print("cannot manually set coordinate systems")
        return    

    @property
    def solution_params(self)-> dict:
        return self.hfss.available_variations.nominal_w_values_dict

    @solution_params.setter
    def solution_params():
        print("Not implemented ability to modify the solution params")


    @property
    def freqs(self):
        f = utils.custom_basis_frequencies_collector(self.hfss, self.setup, self.sweep)
        print(f"FREQS in the HFSS solutions: {f}")
        return f
    
    @freqs.setter
    def freqs():
        print("the frequency are extracted directly from the HFSS project database. You cannot change them.")

    
    
    def _check_coordinate_systems(self):
        allCS = check_coordinate_systems(self.hfss)
        self._coordinate_systems = allCS
    
    def set_coordinate_system(self, CS_name):
        self.hfss.modeler.set_working_coordinate_system(CS_name)
        
        if CS_name == "Global":
            z_shift = 0
        else: 
            my_CS = next(filter(lambda CS: CS.name == CS_name, self.coordinate_systems))
            z_shift = my_CS.z_shift

            if z_shift[-2:] not in ["cm", "mm", "um", "nm"] and z_shift in self.solution_params.keys():
                z_shift = self.solution_params[z_shift]

        self._working_coordinate_system = CoordinateSystem(CS_name, z_shift)    

    def change_CS(self):
        assert len(self.coordinate_systems)==2, " this function only works with exactly 2 coordinate systems"
        name = next(filter(
            lambda name : name != self.working_coordinate_system.name, 
            [CS.name for CS in self.coordinate_systems])
            )    
        self.set_coordinate_system(name)
        return self
    
    def import_3D_fields(self, field_names: Iterable[str], grid, smooth_field=True):
        for name in field_names:
            print(f"extracting {name} field...")
            self.get_3Dfield(name, grid, smooth_field=smooth_field)
        return self
    
    def save_loaded_fields(self, h5path, savename, overwrite=False, attrs={}):
        field_list = []
        if "E" in dir(self):
            field_list.append(self.E)
        if "H" in dir(self):
            field_list.append(self.H)

        save_fields_to_h5(fields = field_list, full_h5path=h5path, savename=savename, overwrite=overwrite, hfss=self.hfss, attrs=attrs)
        return self

        

    def get_3Dfield(self, fieldname, solution_points_grid, smooth_field=True) -> field_classes.Field3D:            
        temp_directory = "temp_for_fields"
        if not os.path.exists(temp_directory):
            os.mkdir(temp_directory)
        temp_points_filepath = os.path.join(temp_directory, "points_file.pts")

        self.make_point_file(solution_points_grid, temp_points_filepath)

        frequency_fields = []

        for f in self.freqs:
            my_sol_params = self.solution_params

            my_sol_params.update(dict(Freq=f"{f*1e-9}GHz", Phase="0deg"))
            print("solution parameters: ", my_sol_params)
            
            filepath = os.path.join(temp_directory, f"_{f}_field.csv")
            
            print(f"exporting frequency {f}")
            # export the field to a temporary file
            self.export_field_to_file(
                fieldname=fieldname, solution_points=temp_points_filepath, 
                destination_file=filepath, solution_parameters=my_sol_params, smooth_field=smooth_field            
            )

            #
            field, _ = utils.get_complex_raw_field_values_from_solution_file(filepath, solution_points_grid.shape)
            frequency_fields.append(field)
            

        all_fields = np.stack(frequency_fields, axis=-1)
        field = field_classes.Field3D(all_fields, self.freqs, solution_points_grid)
        field.__name__ = fieldname
        setattr(self, fieldname, field)

        # remove the temporary files
        shutil.rmtree(temp_directory)

        return field


    def make_point_file(self, grid, filepath, SI_unit="mm"):
        # write file with the points

        if SI_unit == "mm":
            grid = np.array(grid)*1e3

        proj_vars = self.hfss.variable_manager.project_variables
        surface_height = proj_vars.get("$surface_height", 0)

        absolute_grid = copy.deepcopy(grid)
        absolute_grid[-1] = absolute_grid[-1]+surface_height*1e3
        utils.write_point_file(filepath, absolute_grid, SI_unit=SI_unit)


    
    def export_field_to_file(self, fieldname, solution_points, destination_file, solution_parameters=None, raw_field=True, smooth_field=False):
        # by default the solution parameters are those of the class
        if solution_parameters is None:
            solution_parameters = self.solution_params
        
        # create the variation list for the solution
        variation_list = []
        for el, value in solution_parameters.items():
            variation_list.append(el + ":=")
            variation_list.append(value)
        
        # get full paths
        solution_points = os.path.join(os.getcwd(), solution_points)
        destination_file = os.path.join(os.getcwd(), destination_file)

        # get the solution name
        solution = f"{self.setup} : {self.sweep}"

        # extract the field 
        print("extracting the field...")
        self.hfss.post.ofieldsreporter.CalcStack("clear")
        if raw_field:
            self.hfss.post.ofieldsreporter.EnterQty(fieldname)
            if smooth_field:
                self.hfss.post.ofieldsreporter.CalcOp("Smooth")
        else:
            self.hfss.post.ofieldsreporter.CopyNamedExprToStack(fieldname)

        # exporting to file
        print("exporting to file..")
        self.hfss.post.ofieldsreporter.ExportToFile(
            destination_file,
            solution_points,
            solution, 
            variation_list,
            [
                "NAME:ExportOption",
                "IncludePtInOutput:="	, True,
                "RefCSName:="		, self.working_coordinate_system.name,
                "PtInSI:="		, True,
                "FieldInRefCS:="	, False
            ]
        )

