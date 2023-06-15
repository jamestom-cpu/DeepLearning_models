import numpy as np
import pandas as pd
from my_packages.my_pyaedt import my_pyaedt

class HFSS_voltage():
    def __init__(
            self,
            
            hfss,
            voltage_source = 'fw',
            ):
        self.hfss = hfss       
        self.oDesign = self.hfss.odesign
        self.oProject = self.hfss.oproject
        self.oEditor = self.hfss.oeditor
        self.setup_name = "Single"
        self.sweep_name = "InterpolatingHF"
        
        
        #convert the trace thickness into string and mm

        if voltage_source == 'fw':
            self._set_to_forward_wave()
        else:
            self._set_to_total_Voltage()
        
    

        
    def draw_vertical_polyline(self, position_Length, position_Width, name):
        # transform floating point to string in mm
        position_Length_mm = str(position_Length*1e3)+"mm"
        position_Width_mm = str(position_Width*1e3)+"mm"
        


        self.oEditor.CreatePolyline(
            [
                "NAME:PolylineParameters",
                "IsPolylineCovered:="	, True,
                "IsPolylineClosed:="	, False,
                [
                    "NAME:PolylinePoints",
                    [
                        "NAME:PLPoint",
                        "X:="			, position_Width_mm,
                        "Y:="			, position_Length_mm,
                        "Z:="			, "$substrate_thickness"
                    ],
                    [
                        "NAME:PLPoint",
                        "X:="			, position_Width_mm,
                        "Y:="			, position_Length_mm,
                        "Z:="			, "0mm"
                    ]
                ],
                [
                    "NAME:PolylineSegments",
                    [
                        "NAME:PLSegment",
                        "SegmentType:="		, "Line",
                        "StartIndex:="		, 0,
                        "NoOfPoints:="		, 2
                    ]
                ],
                [
                    "NAME:PolylineXSection",
                    "XSectionType:="	, "None",
                    "XSectionOrient:="	, "Auto",
                    "XSectionWidth:="	, "0mm",
                    "XSectionTopWidth:="	, "0mm",
                    "XSectionHeight:="	, "0mm",
                    "XSectionNumSegments:="	, "0",
                    "XSectionBendType:="	, "Corner"
                ]
            ], 
            [
                "NAME:Attributes",
                "Name:="		, name,
                "Flags:="		, "NonModel#",
                "Color:="		, "(143 175 143)",
                "Transparency:="	, 0.7,
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
            ]
        )

    def remove_polylines(self, names):
        oEditor = self.oDesign.SetActiveEditor("3D Modeler")
        selections = ",".join(names)
        oEditor.Delete(
            [
                "NAME:Selections",
                "Selections:="		, selections
            ])


    def calculate_voltage_FieldsReporter(self, line_name, voltage_variable_name):
        oModule = self.oDesign.GetModule("FieldsReporter")
        oModule.CalcStack("clear")
        oModule.EnterQty("E")
        oModule.CalcOp("ScalarY")
        oModule.CalcStack("undo")
        oModule.CalcOp("ScalarZ")
        oModule.CalcOp("Real")
        oModule.EnterLine(line_name)
        oModule.CalcOp("Integrate")
        oModule.EnterQty("E")
        oModule.CalcOp("ScalarZ")
        oModule.CalcOp("Imag")
        oModule.EnterLine(line_name)
        oModule.CalcOp("Integrate")
        oModule.EnterComplex("1 j")
        oModule.CalcOp("*")
        oModule.CalcOp("+")
        oModule.AddNamedExpression(voltage_variable_name, "Fields")
    
    def delete_expressions(self, names):
        if type(names) is not list:
            names = list([names])
        oModule = self.oDesign.GetModule("FieldsReporter")
        oModule.DeleteNamedExpr(names)
    
    def create_fields_report(self, field_report_name, quantity_names):
        if type(quantity_names) is list:
            pass
        else:
            quantity_names = [quantity_names]
        
        property_dict = self.hfss.available_variations.nominal
        property_dict = ["Freq:=", ["All"], "Phase:=", ["0deg"]] + property_dict

        
        
        table_components = [ "X Component:=", "Freq", "Y Component:=", quantity_names]


        oModule = self.oDesign.GetModule("ReportSetup")

        oModule.CreateReport(
            field_report_name, 
            "Fields", "Data Table", f"{self.setup_name} : {self.sweep_name}", [], 
            property_dict, 
            table_components
            )

        self.field_report_name = field_report_name
        self.oModule = oModule
        return oModule
    
    def delete_report(self):
        self.oModule.DeleteReports([self.field_report_name])
    
    def export_table(self, field_report_name, file_name, oModule = None):
        if oModule is None:
            if self.oModule is None:
                return ImportError
            else: 
                oModule = self.oModule 
        
        oModule.ExportToFile(field_report_name, file_name, False)


    def _set_to_total_Voltage(self):
        oModule = self.oDesign.GetModule("Solutions")
        oModule.EditSources(
                [
                    "UseIncidentVoltage:=", False,
                    "IncludePortPostProcessing:=", True,
                    "SpecifySystemPower:=", False
                ],
                [
                    "Name:="	, "Trace_T1",
                    "Terminated:="		, False,
                    "Magnitude:="		, "1V",
                    "Phase:="		, "0deg"
                ],
                [
        			"Name:="		, "Trace_T2",
        			"Terminated:="		, True,
        			"Resistance:="		, "50ohm",
        			"Reactance:="		, "0"
        		]
            )

    def _set_to_forward_wave(self):
        oModule = self.oDesign.GetModule("Solutions")
        oModule.EditSources(
            [
                "UseIncidentVoltage:="	, True,
                "IncludePortPostProcessing:=", False,
                "SpecifySystemPower:="	, False
            ],
            [
                "Name:="		, "trace_T1",
                "Magnitude:="		, "1V",
                "Phase:="		, "0deg"
            ],
            [
                "Name:="		, "trace_T2",
                "Magnitude:="		, "0V",
                "Phase:="		, "0deg"
            ]
        )
        

    def save_project(self, save_name=False):
        if save_name:
            self.oProject.Save()
        else:       
            self.oProject.SaveAs(save_name,True)



def create_integrationlines(vClass, N, name="p"):
    trace_length = vClass.hfss.available_variations.nominal_w_values_dict.get("$trace_length")
    trace_length = my_pyaedt.mmstring2float(trace_length)
    
    x_positions = np.linspace(-1, 1, N)*trace_length/2

    line_names = []
    for ii, x in enumerate(x_positions):
        line_name = f"{name}{ii+1}"
        vClass.draw_vertical_polyline(0, x, name = line_name)
        line_names.append(line_name)
    return line_names

def create_voltage_variables(vClass, line_names):
    voltage_names = []
    for line in line_names:
        voltage_name = f"V_{line}"
        vClass.calculate_voltage_FieldsReporter(line, voltage_name)
        voltage_names.append(voltage_name)
    return voltage_names


def read_HFSS_table(file_path):
    V = pd.read_csv(file_path)
    V.pop('Phase [deg]')
    tab = V.copy()
    f = V.pop('Freq [GHz]').to_numpy()*1e9
    V = V.applymap(transform_csv_string_to_complex)
    V_np = V.to_numpy()

    return -V_np, f

def transform_csv_string_to_complex(string: str):
    # the string is of the kind x + yi
    x = "".join(string.split(" "))
    xj = x.replace("i", "j")
    #returns a "complex" object
    return np.complex(xj)