from typing import Tuple
import pyaedt

from my_packages.my_pyaedt.classes.build_dipole_model import BuildDipoleModel, DipoleModelExcitation
from my_packages.my_pyaedt.classes.build_dipole_model import CreateSimulationEnvironment, AssignSetup
from my_packages.classes.dipole_fields import DFHandler_over_Substrate
from my_packages.classes.model_components import Substrate, UniformEMSpace

def create_model(
        hfss: pyaedt.Hfss, solution_dims: Tuple[float, float, float], 
        solution_space_position : float | Tuple[float, float, float], 
        project_name: str,
        substrate: Substrate,
        dipole_field_handler: DFHandler_over_Substrate, 
        high_density_dims: Tuple[float, float, float], 
        high_density_length:float, 
        high_density_box_position = 0.0,
        minimum_radius_dipole: float = 0.5e-3,
        ANALYZE=False):
    
    substr = dipole_field_handler.substrate
    freqs = dipole_field_handler.f
    
    smSpace = CreateSimulationEnvironment(
        hfss, dimensions=solution_dims, position=solution_space_position, 
        max_length=high_density_length, solution_type="terminal"
        ).create_new_project(project_name)

    board_name = "BoardBox"
    # surface_CS_name = "BoardSurface_CS"

    bmodel = BuildDipoleModel(hfss, substrate, board_object_name=board_name, sheet_boundary_backing=True, group_name_for_model="rModel")
    bmodel.create_passive_model(use_variable_names = True)

    smSpace.create_solution_space().set_boundary_condition().set_max_mesh_length(
        box_dims=high_density_dims, 
        box_position=high_density_box_position
        )
    ast = AssignSetup(hfss, solution_frequency=freqs, basis_order = 1).define_setup().make_sweep()
    print("FREQ", ast.solution_frequency)

    ## assign excitations
    source_cntrl = DipoleModelExcitation(hfss, substr, dipole_field_handler, minimum_radius_dipole=minimum_radius_dipole*1e3)
    source_cntrl.setup_dipoles().set_dipole_moments()

    bmodel.set_relative_CS()
    hfss.set_active_design(project_name)
    if ANALYZE:
        hfss.analyze_setup(ast.setup_name)
    return {"build_model": bmodel, "simulation_space": smSpace, "setup": ast, "source_control": source_cntrl}