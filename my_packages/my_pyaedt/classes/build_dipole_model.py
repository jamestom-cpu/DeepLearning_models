import numpy as np

from copy import copy
from dataclasses import dataclass, field
from typing import Union, Tuple, List



from my_packages.classes import dipole_array, dipole_fields
from my_packages.classes.model_components import Substrate
from .excitations import BuildPhysicalElectricDipole

from ..funcs_for_building_the_model import *

import pyaedt



@dataclass
class AssignSetup():
    hfss: pyaedt.Hfss
    solution_frequency:float = 1e9
    DeltaS:float = 0.1
    Max_passes:int = 80
    converged_patience:int = 3
    refinement_percent: int=30
    setup_name:str ="Default_Setup"
    sweep_name: str = "Default_Sweep"
    freq_range: Tuple = (250e6, 1e9)
    step_value: float = 250e9
    define_range: bool = False
    basis_order: int = 1

    
    def define_setup(self):

        if self.setup_name in self.hfss.setup_names:
            # maybe the setup has been deleted but the project wasn't saved
            self.hfss.save_project()
        
        ii=1
        original_setup_name = self.setup_name
        while self.setup_name in self.hfss.setup_names:
            print(f"name already in use: {self.setup_name}")
            # if the setup is still within the setup_names, then create a new name
            self.hfss.save_project()
            self.setup_name = original_setup_name+f"_{ii}"
            ii+=1
            print(f"added_name: {self.setup_name}")
        if isinstance(self.solution_frequency, float):
            f = self.solution_frequency
            print(f)
        else:
            self.solution_frequency = sorted(list(self.solution_frequency))
            f = self.solution_frequency[-1]
            print(f)
        define_setup(
            self.hfss, solution_frequency=f, DeltaS=self.DeltaS, 
            Max_passes=self.Max_passes, converged_patience=self.converged_patience, refinement_percent=self.refinement_percent,
            name = self.setup_name, basis_order = self.basis_order)
        return self

    def make_sweep(self):
        if self.setup_name in self.hfss.get_sweeps(self.setup_name):
            # maybe the setup has been deleted but the project wasn't saved
            self.hfss.save_project()
        
        ii=1
        original_sweep_name = self.sweep_name
        while self.sweep_name in self.hfss.get_sweeps(self.setup_name):
            print(f"name already in use: {self.sweep_name}")
            # if the setup is still within the setup_names, then create a new name
            self.hfss.save_project()
            self.sweep_name = original_sweep_name+f"_{ii}"
            ii+=1
            print(f"added_name: {self.sweep_name}")
        
        make_sweep_HFSS(
            hfss= self.hfss, setup_name=self.setup_name, sweep_name=self.sweep_name, f=self.solution_frequency, freq_range=self.freq_range, 
            step_value=self.step_value, define_range=False)
        return self

@dataclass
class CreateSimulationEnvironment():
    hfss: pyaedt.Hfss
    solution_space_name:str = "MySolutionSpace"
    dimensions: Tuple = (5e-2, 5e-2, 2.5e-2)
    position: Union[str, Tuple, float] = "xycentered"
    use_variable_names: bool = False
    variable_names: List = field(default_factory=lambda: ["xdim", "ydim", "zdim"])
    material_name: str ="vacuum"
    boundary_type: str ="Radiation"
    max_length:float = 3e-3
    refine_inside: bool =True
    solution_type: str = "modal"

    def create_new_project(self, project_name, **kargs):
        create_new_project(self.hfss, project_name, **kargs)
        set_solution_type(self.hfss, self.solution_type)
        return self
    
    def create_box(self, position, dims, box_name):
        if isinstance(position, str):
            if position == "centered":
                position = tuple([-x/2 for x in dims]) 
            elif position == "xycentered": 
                position = tuple([-x/2 for x in dims[:-1]]+[0])
            else:
                raise Exception(f"The String {position} is not recognized")  
        if isinstance(position, float):
            position = tuple([-x/2 for x in dims[:-1]]+[position]) 
        create_centered_box(
            self.hfss, *dims, box_object_name=box_name, 
            use_variable_names=False)
        change_boxobject_position(self.hfss, name=box_name, value=position)
        return self
    
    def create_solution_space(self):
        if isinstance(self.position, str):
            if self.position == "centered":
                self.position = tuple([-x/2 for x in self.dimensions]) 
            elif self.position == "xycentered": 
                self.position = tuple([-x/2 for x in self.dimensions[:-1]]+[0])
            else:
                raise Exception(f"The String {self.position} is not recognized")  
        if isinstance(self.position, float):
            self.position = tuple([-x/2 for x in self.dimensions[:-1]]+[self.position]) 
        create_centered_box(
            self.hfss, *self.dimensions, box_object_name=self.solution_space_name, 
            use_variable_names=self.use_variable_names, variable_names=self.variable_names)
        change_boxobject_position(self.hfss, name=self.solution_space_name, value=self.position)
        return self
    
    def change_space_material(self):
        change_material(self.hfss, self.solution_space_name, self.material_name)
        return

    
    def set_boundary_condition(self):
        set_boundary_conditions(self.hfss, self.solution_space_name, boundary_type=self.boundary_type)
        return self
    
    def set_max_mesh_length(self, box_dims = None, box_position = "xycentered", max_length = None, box_name = "DenseMeshBox", **kargs):
        # if box_dims is None, then the max_length is set for the solution space
        if box_dims is not None:
            self.create_box(dims=box_dims, position=box_position, box_name=box_name)
            make_non_model(self.hfss, box_name)
        target_object = self.solution_space_name if box_dims is None else box_name
        if max_length is None:
            max_length = self.max_length
        set_mesh_max_length(
            self.hfss, max_length=max_length, target_object_name=target_object, 
            inside_selection=self.refine_inside,**kargs)
        return self
 
    def delete_project(self):
        self.hfss.delete_design()
        return self

@dataclass
class BuildDipoleModel():
    hfss:pyaedt.Hfss
    substrate: Substrate
    board_object_name:str = "DipoleArray_Box"
    ground_sheet_name: str = "GND"
    ground_sheet_thickness: float = 1e-5
    relative_CS_name: str = "BoardSurface_CS"
    group_name_for_model: str = "myModel"
    sheet_boundary_backing: bool = False

    def add_project_variable(self, name:str, value:float):
        add_project_variable(self.hfss, name, value)
        return self

    def create_board_passive_geometry_structure(self, **kargs):
        self.substrate_box_name = self.board_object_name
        self.sheet_name = self.ground_sheet_name
        create_centered_box(
            hfss= self.hfss, box_dim_z=self.substrate.thickness, 
            box_dim_x=self.substrate.x_size, box_dim_y= self.substrate.y_size, 
            box_object_name= self.board_object_name, **kargs)
        create_sheet_backing_on_box(
            self.hfss, self.substrate_box_name, self.sheet_name, 
            self.ground_sheet_thickness, thicken_sheet=(not self.sheet_boundary_backing))
        return self
    
    def change_object_material(self, model_object_name, material_name):
        change_material(self.hfss, model_object_name, material_name)
        return self
    
    def set_relative_CS(self, z_shift=None):
        if z_shift is None:
            z_shift = self.substrate.thickness
        set_relative_CS_on_substrate_surface(
            self.hfss, z_shift, name=self.relative_CS_name)
        self.hfss.save_project()
        return self
    
    def create_passive_model(self, **kargs):
        self.create_board_passive_geometry_structure(**kargs)
        self.change_object_material(
            self.substrate_box_name, self.substrate.material_name)
        if not self.sheet_boundary_backing:
            self.change_object_material(self.sheet_name, "pec")
        else:
            print("assigning PEC boundary")
            set_boundary_conditions(self.hfss, self.sheet_name, "PEC", boundary_name="GND_Boundary")
        collect_into_group(
            self.hfss, [self.substrate_box_name, self.sheet_name], 
            self.group_name_for_model)
        return self

@dataclass
class DipoleModelExcitation():
    hfss: pyaedt.Hfss
    substrate: Substrate
    dfield: dipole_fields.Dipole_Field
    minimum_radius_dipole: float = 1e-9 
    dipole_base_name: str = "Dipole"
    relative_CS_name: str = "BoardSurface_CS"
    phdipole_length: float = 0.1e-3
    phdipole_width: float = 0.1e-3/20
    phdipole_correction_factor: float = 0.95
    build_physical_E_dipoles: bool = False
    phdipoles_params: dict = field(default_factory=dict)
    dipole_names: list = field(default_factory=list)
    phdipoles_names: list = field(default_factory=list)

    def setup_dipoles(self, physical_E_dipoles=False, **kwargs):       
        if physical_E_dipoles:
            E_array = self.dfield.dipole_array.get_single_type_array("Electric")
            H_array = self.dfield.dipole_array.get_single_type_array("Magnetic")

            if E_array.N_dipoles>0:
                self.setup_dipoles_physical(E_array, **kwargs) 
            if H_array.N_dipoles>0:
                self.setup_dipoles_incident_wave(H_array)
        else:
            self.setup_dipoles_incident_wave(self.dfield.dipole_array, **kwargs)
        return self


    def setup_dipoles_physical(self, darray: dipole_array.DipoleSourceArray=None, **other_params):
        if darray is None:
            darray = self.dfield.dipole_array
        assert all([d.type == "Electric" for d in darray.dipoles]), "All dipoles must be electric dipoles"
        self.phdipoles_names = [f"{d.type}{self.dipole_base_name}_{ii}" for ii, d in enumerate(darray.dipoles)]


        other_params = self.phdipoles_params | other_params

        for ii, d in enumerate(darray.dipoles):
            BuildPhysicalElectricDipole(
                self.hfss, position=d.r0, orientation=[d.theta, d.phi], 
                length=self.phdipole_length, width=self.phdipole_width,
                name= self.phdipoles_names[ii],
                relative_CS_name=self.relative_CS_name, current_source_name=self.phdipoles_names[ii],
                **other_params).build()





    def setup_dipoles_incident_wave(self, darray: dipole_array.DipoleSourceArray=None, add_extra_gap_in_z_direction = 0, **kwargs):       
        if darray is None:
            darray = self.dfield.dipole_array

        zero_plane_height = self.substrate.thickness + add_extra_gap_in_z_direction
        base_names = [f"{d.type}{self.dipole_base_name}" for d in darray.dipoles]
        d_types = [d.type for d in darray.dipoles]



        self.dipole_names = setup_dipoles(
            self.hfss, darray, self.minimum_radius_dipole, 
            base_name=base_names, type=d_types, 
            relative_height=zero_plane_height)
        return self
    
    def set_dipole_moments(self, f_index=0, dipole_names=None, physical_E_dipoles=False):
        if dipole_names is None:
            my_dipole_names = self.dipole_names
        else:
            my_dipole_names = dipole_names
        
        # for magnetic dipoles, HFSS considers the moments in magnetic currents
    
        # the class automatically transforms the magnetic dipole moment in the I*A form.
        # this is true wheater or not the variable magnetic current source is set to True. 
        # Indeed, this variable is specifying weather the input moments are meant as I*A or magnetic current*Dl,
        # internally it is then converted to I*A
        M = np.array(self.dfield.dipole_array.dipole_moments)
        dips = self.dfield.dipole_array.dipoles
        factor = [
            1j*2*np.pi*self.dfield.f*self.dfield.mu0*self.dfield.mu_r 
            if d.type=="Magnetic" else np.ones_like(self.dfield.f) for d in dips
            ]
        
        factor = np.stack(factor, axis=0)
        scaled_moments = factor*M

        if physical_E_dipoles:
            factor = [
                self.phdipole_correction_factor/self.phdipole_length*np.ones_like(self.dfield.f) 
                if d.type=="Electric" else np.ones_like(self.dfield.f) for d in dips
            ]
            factor = np.stack(factor, axis=0)
            scaled_moments = factor*scaled_moments

        units=""
        set_dipole_moments(self.hfss, scaled_moments[:, f_index], my_dipole_names, units=units)
        


        if dipole_names == None:
            moms = self.dfield.dipole_array.get_single_type_array("Electric").M
            current_values = (self.phdipole_correction_factor*np.asarray(moms)/self.phdipole_length).squeeze()
            set_dipole_moments(self.hfss, current_values, self.phdipoles_names, units="A")
            return self
    
    def assign_dipoles_to_box(self, boxname, active_design=None):
        if type(boxname) is not list:
            boxname = [boxname] 
        if active_design is not None:
            self.hfss.set_active_design(active_design)
        assign_dipoles_to_object(self.hfss, self.dipole_names, boxname)
        return self
    
    def remove_assignment(self, active_design=None):
        if active_design is not None:
            self.hfss.set_active_design(active_design)
        remove_dipole_assignment(self.hfss, self.dipole_names)
        return self

        

