from copy import copy
import pyaedt
import numpy as np

from my_packages.classes import dipole_array
from typing import Tuple, Union, Iterable

def create_new_project(hfss: pyaedt.Hfss, name: str, override=True, project_type = "Terminal")-> None:
    hfss.save_project()
    if name in hfss.design_list and override:
        success = hfss.delete_design(name)
        assert success, "Was not able to delete design"
        hfss.save_project()

    hfss.insert_design(name, project_type)

def set_solution_type(hfss: pyaedt.Hfss, solution_type: str = "terminal"):
    if solution_type == "terminal":
        full_name = "HFSS Terminal Network"
    elif solution_type == "modal":
        full_name = "HFSS Modal Network"
    
    hfss.odesign.SetSolutionType(full_name, 
	    [
            "NAME:Options",
            "EnableAutoOpen:="	, False
        ])


def setup_dipoles(hfss, dipole_array: dipole_array.DipoleSourceArray, radius=1e-5, base_name="dipole", type="magnetic", relative_height = 0):
    """the radius is specified in mm"""
    unit_vectors = [dipole.unit_p for dipole in dipole_array.dipoles]
    positions = copy(dipole_array.r0)
    positions[:,-1] = positions[:, -1] + np.full_like(positions[:,-1], relative_height)

    if isinstance(base_name, str):
        base_name = [base_name]*dipole_array.N_dipoles

    if isinstance(type, str):
        type = [type]*dipole_array.N_dipoles
    
    assert len(base_name)==len(type) and len(base_name)==dipole_array.N_dipoles, f"""the lengths 
    of type and base name should be equal to {dipole_array.N_dipoles}. Instead, the len of `base name` is {len(base_name)}
     and the len of `type` is {len(type)}"""

    names = [f"{base_name[ii]}{ii+1}" for ii in range(dipole_array.N_dipoles)]


    print(names)
    print(positions)
    print(unit_vectors)

    boundary_module = hfss.get_module("BoundarySetup")


    for ii in range(dipole_array.N_dipoles):
        boundary_module.AssignHertzianDipoleWave(
            [
                f"NAME:{names[ii]}",
                "IsCartesian:="		, True,
                "EoX:="			, str(unit_vectors[ii][0]),
                "EoY:="			, str(unit_vectors[ii][1]),
                "EoZ:="			, str(unit_vectors[ii][2]),
                "kX:="			, "0",
                "kY:="			, "0",
                "kZ:="			, "1",
                "OriginX:="		, str(positions[ii][0]*1e3)+"mm",
                "OriginY:="		, str(positions[ii][1]*1e3)+"mm",
                "OriginZ:="		, str(positions[ii][2]*1e3)+"mm",
                "SphereRadius:="	, format(radius, ".8f")+"mm",
                "IsElectricDipole:="	, type[ii]=="Electric"
            ]
        )
    return names

def set_dipole_moments(hfss, moments, dipole_names, units="Vm"):
    solution_module = hfss.get_module("Solutions") 
    for M, d_name in zip(moments, dipole_names):
        print(f"{np.real(M)}Vm + {np.imag(M)}i Vm")
        print(d_name)
        solution_module.EditSources(
            [
                "FieldType:="		, "TotalFields",
                "IncludePortPostProcessing:=", False,
                "SpecifySystemPower:="	, False
            ])
        
        magnitude = np.abs(M)
        phase = np.angle(M, deg=True)

        print(f"{magnitude}{units}")
        print(f"{phase}deg")
        
        solution_module.EditSources(
            [
                "Name:="		, d_name,
                "Magnitude:="		, f"{magnitude}{units}",
                "Phase:="		, f"{phase}deg"
            ]
        )




def change_boxobject_position(hfss: pyaedt.Hfss, name:str, value: Tuple):
    oEditor = hfss.odesign.SetActiveEditor("3D Modeler")
    oEditor.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:Geometry3DCmdTab",
                [
                    "NAME:PropServers", 
                    f"{name}:CreateBox:1"
                ],
                [
                    "NAME:ChangedProps",
                    [
                        "NAME:Position",
                        "X:="			, f"{value[0]*1e3}mm",
                        "Y:="			, f"{value[1]*1e3}mm",
                        "Z:="			, f"{value[2]*1e3}mm"
                    ]
                ]
            ]
        ])

def make_non_model(hfss:pyaedt.Hfss, object_name:str):
    oEditor = hfss.odesign.SetActiveEditor("3D Modeler")
    oEditor.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:Geometry3DAttributeTab",
                [
                    "NAME:PropServers", 
                    object_name
                ],
                [
                    "NAME:ChangedProps",
                    [
                        "NAME:Model",
                        "Value:="		, False
                    ]
                ]
            ]
        ])
    
def add_project_variable(hfss, name:str, value):
    oDesign = hfss.odesign
    oDesign.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:LocalVariableTab",
                [
                    "NAME:PropServers", 
                    "LocalVariables"
                ],
                [
                    "NAME:NewProps",
                    [
                        "NAME:"+name,
                        "PropType:="		, "VariableProp",
                        "UserDef:="		, True,
                        "Value:="		, f"{value*1e3}mm"
                    ]
                ]
            ]
        ])


def create_centered_box(
    hfss, box_dim_x, box_dim_y, box_dim_z, 
    box_object_name = "Box", 
    variable_names=["x_dimension", "y_dimension", "z_dimension"],
    use_variable_names = True
    ):
    
    if use_variable_names:
        for name, value in zip(variable_names, [box_dim_x, box_dim_y, box_dim_z]):
            add_project_variable(hfss, name, value)
        
    xsize = variable_names[0] if use_variable_names else str(box_dim_x*1e3)+"mm"
    ysize = variable_names[1] if use_variable_names else str(box_dim_y*1e3)+"mm"
    zsize = variable_names[2] if use_variable_names else str(box_dim_z*1e3)+"mm"

    oEditor = hfss.odesign.SetActiveEditor("3D Modeler")    
    oEditor.CreateBox(
        [
            "NAME:BoxParameters",
            "XPosition:="		, f"-{xsize}/2",
            "YPosition:="		, f"-{ysize}/2",
            "ZPosition:="		, "0mm",
            "XSize:="		, xsize,
            "YSize:="		, ysize,
            "ZSize:="		, zsize
        ], 
        [
            "NAME:Attributes",
            "Name:="		, box_object_name,
            "Flags:="		, "",
            "Color:="		, "(143 175 143)",
            "Transparency:="	, 0,
            "PartCoordinateSystem:=", "Global",
            "UDMId:="		, "",
            "MaterialValue:="	, "\"vacuum\"",
            "SurfaceMaterialValue:=", "\"\"",
            "SolveInside:="		, True,
            "ShellElement:="	, False,
            "ShellElementThickness:=", "0mm",
            "IsMaterialEditable:="	, True,
            "UseMaterialAppearance:=", False,
            "IsLightweight:="	, False
        ])


    
def create_sheet_backing_on_box(hfss, box_name, sheet_name="GND", sheet_thickness=1e-5, thicken_sheet=True):
    oEditor = hfss.odesign.SetActiveEditor("3D Modeler") 

    oEditor.CreateObjectFromFaces(
	[
		"NAME:Selections",
		"Selections:="		, box_name,
		"NewPartsModelFlag:="	, "Model"
	], 
	[
		"NAME:Parameters",
		[
			"NAME:BodyFromFaceToParameters",
			"FacesToDetach:="	, [8]
		]
	], 
	[
		"CreateGroupsForNewObjects:=", False
	])
    
    automatic_name_for_PEC = box_name + "_ObjectFromFace1"
    change_model_object_name(hfss, automatic_name_for_PEC, sheet_name)
    if thicken_sheet:
        oEditor.ThickenSheet(
            [
                "NAME:Selections",
                "Selections:="		, sheet_name,
                "NewPartsModelFlag:="	, "Model"
            ], 
            [
                "NAME:SheetThickenParameters",
                "Thickness:="		, f"{sheet_thickness*1e3}mm",
                "BothSides:="		, False,
                [
                    "NAME:ThickenAdditionalInfo",
                    [
                        "NAME:ShellThickenDirectionInfo",
                        "SampleFaceID:="	, 36,
                        "ComponentSense:="	, True,
                    ]
                ]
            ]
    )

def change_model_object_name(hfss, old_name, new_name):
    oEditor = hfss.odesign.SetActiveEditor("3D Modeler") 
    oEditor.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:Geometry3DAttributeTab",
                [
                    "NAME:PropServers", 
                    old_name
                ],
                [
                    "NAME:ChangedProps",
                    [
                        "NAME:Name",
                        "Value:="		, new_name
                    ]
                ]
            ]
        ])

def change_material(hfss, model_object_name, material_name):
    oEditor = hfss.odesign.SetActiveEditor("3D Modeler") 
    oEditor.ChangeProperty(
	[
		"NAME:AllTabs",
		[
			"NAME:Geometry3DAttributeTab",
			[
				"NAME:PropServers", 
				model_object_name
			],
			[
				"NAME:ChangedProps",
				[
					"NAME:Material",
					"Value:="		, f"\"{material_name}\""
				]
			]
		]
	])

def set_relative_CS_on_substrate_surface(hfss, board_height, name = "BoardSurface_CS"):
    oEditor = hfss.odesign.SetActiveEditor("3D Modeler") 
    oEditor.CreateRelativeCS(
	[
		"NAME:RelativeCSParameters",
		"Mode:="		, "Axis/Position",
		"OriginX:="		, "0mm",
		"OriginY:="		, "0mm",
		"OriginZ:="		, f"{board_height*1e3}mm",
		"XAxisXvec:="		, "1mm",
		"XAxisYvec:="		, "0mm",
		"XAxisZvec:="		, "0mm",
		"YAxisXvec:="		, "0mm",
		"YAxisYvec:="		, "1mm",
		"YAxisZvec:="		, "0mm"
	], 
	[
		"NAME:Attributes",
		"Name:="		, name
	])

def collect_into_group(hfss, group_parts_name_list, group_name="my_group"):
    
    oEditor = hfss.odesign.SetActiveEditor("3D Modeler") 
    oEditor.CreateGroup(
        [
            "NAME:GroupParameter",
            "ParentGroupID:="	, "Model",
            "Parts:="		, stringify_name_list(group_parts_name_list),
            "SubmodelInstances:="	, "",
            "Groups:="		, ""
        ])
    oEditor.ChangeProperty(
        [
            "NAME:AllTabs",
            [
                "NAME:Attributes",
                [
                    "NAME:PropServers", 
                    "Group1"
                ],
                [
                    "NAME:ChangedProps",
                    [
                        "NAME:Name",
                        "Value:="		, group_name
                    ]
                ]
            ]
        ])

def stringify_name_list(name_list):
    return ",".join(name_list)
# 




def assign_dipoles_to_object(hfss, dipole_names, object_names):
    oModule = hfss.get_module("BoundarySetup") 
    for name in dipole_names:
        oModule.ReassignBoundary(
            [
                f"NAME:{name}",
                "Objects:="		, object_names
            ])



def remove_dipole_assignment(hfss, dipole_names):
    oModule = hfss.get_module("BoundarySetup") 
    for name in dipole_names:
        oModule.ReassignBoundary(
        [
            f"NAME:{name}"
        ])


def set_boundary_conditions(hfss: pyaedt.Hfss, target_object_name: str, boundary_type="Radiation", boundary_name="BoundaryCondition", infinite_ground_plane=False):
    oModule = hfss.odesign.GetModule("BoundarySetup")
    if boundary_type == "PEC":
        oModule.AssignPerfectE(
        [
            f"NAME:{boundary_name}",
            "Objects:="		, [target_object_name],
            "InfGroundPlane:="	, infinite_ground_plane
        ])

    if boundary_type == "Radiation":
        oModule.AssignRadiation(
            [
                f"NAME:{boundary_name}",
                "Objects:="		, [target_object_name]
            ])

def set_mesh_max_length(hfss: pyaedt.Hfss, max_length: float, target_object_name:str, inside_selection=True, settings_name=None):
    if settings_name is None:
        settings_name = "MaxLengthMesh_"+ target_object_name
    oModule = hfss.odesign.GetModule("MeshSetup")
    oModule.AssignLengthOp(
        [
            f"NAME:{settings_name}",
            "RefineInside:="	, inside_selection,
            "Enabled:="		, True,
            "Objects:="		, [target_object_name],
            "RestrictElem:="	, False,
            "NumMaxElem:="		, "1000",
            "RestrictLength:="	, True,
            "MaxLength:="		, f"{max_length*1e3}mm"
        ])

def make_sweep_HFSS(hfss:pyaedt.Hfss, setup_name:str, f: Union[float, Iterable], sweep_name="MySweep", freq_range=(250e6, 1e9), step_value=250e6, define_range=False):
    oModule = hfss.odesign.GetModule("AnalysisSetup")

    if define_range:
        oModule.InsertFrequencySweep(setup_name, 
        [
            f"NAME:{sweep_name}",
            "IsEnabled:="		, True,
            "RangeType:="		, "LinearStep",
            "RangeStart:="		, f"{freq_range[0]*1e-9}GHz",
            "RangeEnd:="		, f"{freq_range[1]*1e-9}GHz",
            "RangeStep:="		, f"{step_value*1e-9}GHz",
            "Type:="		, "Discrete",
            "SaveFields:="		, True,
            "SaveRadFields:="	, False
        ])
        return

    f_list_for_HFSS = []
    if isinstance(f, float):
        f = [f]
        f0 = f
    if len(f)==1:
        f0=f[0]
    if len(f)>1:
        for ff in f[1:]:
            f_list_for_HFSS.append(
                [
				"NAME:Subrange",
				"RangeType:="		, "SinglePoints",
				"RangeStart:="		, f"{ff*1e-6}MHz",
				"RangeEnd:="		, f"{ff*1e-6}MHz",
				"SaveSingleField:="	, False
			    ]
            )

        f_list_for_HFSS = [[
			"NAME:SweepRanges",
			*f_list_for_HFSS
		]]
        f0 = f[0]

    argument_list = [
        f"NAME:{sweep_name}",
        "IsEnabled:="		, True,
        "RangeType:="		, "SinglePoints",
        "RangeStart:="		, f"{f0*1e-6}MHz",
        "RangeEnd:="		, f"{f0*1e-6}MHz",
        "SaveSingleField:="	, False,
        "Type:="		, "Discrete",
        "SaveFields:="		, True,
        "SaveRadFields:="	, False,
    ]

    for extra in f_list_for_HFSS:
        argument_list.insert(-6, extra)
    
    print(argument_list)

    oModule.InsertFrequencySweep(setup_name, argument_list)
    



def define_setup(hfss:pyaedt.Hfss, solution_frequency=1e9, DeltaS=0.1, Max_passes=80, converged_patience=3, refinement_percent=30, name="Default_Setup", basis_order=1):
    oModule = hfss.odesign.GetModule("AnalysisSetup")
    oModule.InsertSetup("HfssDriven", 
        [
            f"NAME:{name}",
            "SolveType:="		, "Single",
            "Frequency:="		, f"{solution_frequency*1e-9}GHz",
            "MaxDeltaE:="		, DeltaS,
            "MaximumPasses:="	, Max_passes,
            "MinimumPasses:="	, 1,
            "MinimumConvergedPasses:=", converged_patience,
            "PercentRefinement:="	, refinement_percent,
            "IsEnabled:="		, True,
            [
                "NAME:MeshLink",
                "ImportMesh:="		, False
            ],
            "BasisOrder:="		, basis_order,
            "DoLambdaRefine:="	, True,
            "DoMaterialLambda:="	, True,
            "SetLambdaTarget:="	, False,
            "Target:="		, 0.3333,
            "UseMaxTetIncrease:="	, False,
            "DrivenSolverType:="	, "Auto Select Direct/Iterative",
            "EnhancedLowFreqAccuracy:=", False,
            "SaveRadFieldsOnly:="	, False,
            "SaveAnyFields:="	, True,
            "CacheSaveKind:="	, "Delta",
            "ConstantDelta:="	, "0s",
            "IESolverType:="	, "Auto",
            "LambdaTargetForIESolver:=", 0.15,
            "UseDefaultLambdaTgtForIESolver:=", True,
            "IE Solver Accuracy:="	, "Balanced",
            "InfiniteSphereSetup:="	, ""
        ])
    


def set_CS_at_sheet_center(hfss: pyaedt.Hfss, sheet_name: str, sheet_id: int, center_position: Tuple[float,float, float], CS_name="CS_SheetCenter"):

    oEditor = hfss.odesign.SetActiveEditor("3D Modeler")
    oEditor.CreateObjectCS(
	[
		"NAME:ObjectCSParameters",
		[
			"NAME:Origin",
			"IsAttachedToEntity:="	, True,
			"EntityID:="		, sheet_id,
			"FacetedBodyTriangleIndex:=", -1,
			"TriangleVertexIndex:="	, -1,
			"hasXYZ:="		, True,
			"PositionType:="	, "FaceCenter",
			"UParam:="		, 0,
			"VParam:="		, 0,
			"XPosition:="		, str(center_position[0]),
			"YPosition:="		, str(center_position[1]),
			"ZPosition:="		, str(center_position[2])
		],
		"MoveToEnd:="		, False,
		"ReverseXAxis:="	, False,
		"ReverseYAxis:="	, False,
		[
			"NAME:xAxis",
			"DirectionType:="	, "AbsoluteDirection",
			"EdgeID:="		, -1,
			"FaceID:="		, -1,
			"xDirection:="		, "1",
			"yDirection:="		, "0",
			"zDirection:="		, "0",
			"UParam:="		, 0,
			"VParam:="		, 0
		],
		[
			"NAME:yAxis",
			"DirectionType:="	, "AbsoluteDirection",
			"EdgeID:="		, -1,
			"FaceID:="		, -1,
			"xDirection:="		, "0",
			"yDirection:="		, "1",
			"zDirection:="		, "0",
			"UParam:="		, 0,
			"VParam:="		, 0
		]
	], 
	[
		"NAME:Attributes",
		"Name:="		, CS_name,
		"PartName:="		, sheet_name,
	])