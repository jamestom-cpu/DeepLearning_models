from dataclasses import dataclass, field
from typing import Tuple
from pyaedt import Hfss


from ..funcs_for_building_the_model import *
from .solved import check_coordinate_systems


def _change_rectangle_coordinate_system(hfss:Hfss, name:str, CS_name:str):
	oEditor = hfss.odesign.SetActiveEditor("3D Modeler")

	oEditor.ChangeProperty(
		[
			"NAME:AllTabs",
			[
				"NAME:Geometry3DCmdTab",
				[
					"NAME:PropServers", 
					f"{name}:CreateRectangle:1"
				],
				[
					"NAME:ChangedProps",
					[
						"NAME:Coordinate System",
						"Value:="		, CS_name
					]
				]
			]
		])

@dataclass
class BuildPhysicalElectricDipole():
    hfss: Hfss
    length: float
    width: float 
    orientation: Tuple[float, float]
    position: Tuple[float, float, float]
    relative_CS_name: str = "BoardSurface_CS"
    current_source_name: str = "DipoleSource"
    name: str = "Dipole"
    set_variable_names: bool = False
    variable_names: dict = field(default_factory=lambda: {"length": "dipole_l", "width": "dipole_w"})
    dipole_color: list= field(default_factory=lambda: [0, 0, 0])

    def __post_init__(self):
        check_coordinate_systems(self.hfss)
        self.set_coordinate_system(self.relative_CS_name)
        


    @property
    def coordinate_systems(self):
        return self._coordinate_systems

    @coordinate_systems.setter
    def coordinate_systems(self, _):
        print("cannot manually set coordinate systems")
        return  
    
    def set_coordinate_system(self, CS_name):
        self.hfss.modeler.set_working_coordinate_system(CS_name)

    
    def add_project_variable(self, name:str, value:float):
        add_project_variable(self.hfss, name, value)
        return self
    
    def _create_body(self):
        if self.set_variable_names:
            self.add_project_variable(self.variable_names["length"], self.length)
            self.add_project_variable(self.variable_names["width"], self.width)
        
        position = np.asarray(self.position)*1e3
        width = np.asarray(self.width)*1e3
        length = np.asarray(self.length)*1e3

        corner_position = [position[0], position[1] - width/2, position[2]-length/2]
        if self.set_variable_names:
            corner_position = [position[0], f"{position[1]}-{self.variable_names['width']}/2", f"{position[2]}-{self.variable_names['length']}/2"]

        rect = self.hfss.modeler.primitives.create_rectangle(
            csPlane="YZ", position=corner_position, 
            dimension_list=[width, length], 
            name=self.name)
        # update with latest name
        self.name = rect.name

        # change the coordinate system
        _change_rectangle_coordinate_system(self.hfss, self.name, self.relative_CS_name)        
        # set color to black
        self.hfss.modeler.primitives[self.name].color = self.dipole_color
        # create a local coordinate system on the center of the rectangle
        self.CS_name = self.name + "_CS" 
        set_CS_at_sheet_center(self.hfss, self.name, rect.faces[0].id, center_position=self.position, CS_name=self.CS_name)
        return rect

    def _rotate_to_correct_orientation(self, rect):
        # rotate the dipole
        rect.rotate("Y", angle=self.orientation[0], unit="rad")
        rect.rotate("Z", angle=self.orientation[1], unit="rad")


    def _assign_perfect_e(self):
        self.hfss.assign_perfecte_to_sheets(self.name)
    
    def _assign_current(self):
        self.hfss.assign_current_source_to_sheet(self.name, self.hfss.AxisDir.ZNeg, self.current_source_name)

    def _create(self):
        rect = self._create_body()
        self._assign_perfect_e()
        self._assign_current()
        self._rotate_to_correct_orientation(rect)
        return self
    
    # alias for _create
    def build(self):
        return self._create()

