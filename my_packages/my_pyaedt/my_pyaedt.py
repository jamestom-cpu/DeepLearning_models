import os

import numpy as np

import pyaedt
from pyaedt.generic.general_methods import remove_project_lock

from my_packages.my_pyaedt import utils

def start_hfss(project_name, active_design=None, non_graphical=False, new_Thread=False, return_desktop=False, **kwargs):
    ## STARTUP
    # set to non graphical (precedence goes to the environment variable)
    # non_graphical = os.getenv("PYAEDT_NON_GRAPHICAL", str(non_graphical)).lower() in ("true", "1", "t")
    non_graphical=non_graphical

    # start HFSS 
    desktopVersion = "2022.2"
    NewThread = new_Thread
    desktop = pyaedt.Desktop(desktopVersion, non_graphical=non_graphical, new_desktop_session=NewThread)

    # open project 
    try:
        remove_project_lock(project_name)
        hfss = pyaedt.Hfss(project_name, active_design, **kwargs)
    except:
        hfss = pyaedt.Hfss(designname=active_design, **kwargs)

    if return_desktop:
        return hfss, desktop

    return hfss
    #################################################


def export_HFSS_field(
    f, which_fields, field_paths, 
    solution_parameters, grid, hfss, 
    setup, sweep, points_filename, 
    raw_field=True, smooth_field=False
    ):

    solution_parameters.update(dict(Freq=f"{f*1e-6}MHz", Phase="0deg"))

    my_fields = {}

    #export field values from HFSS
    for field, path in zip(which_fields, field_paths):
        export_field_to_file(
            fieldname=field, points_filename=points_filename, solution_parameters=solution_parameters, 
            destination_filename=path, hfss=hfss,
            setup=setup, sweep=sweep, raw_field=raw_field, smooth_field=smooth_field
            )

        # open field file from folder
        field3D, _ = utils.get_complex_raw_field_values_from_solution_file(path, np.array(grid).shape)
        my_fields.update({field: field3D})

        os.remove(path)

    return my_fields 


def export_field_to_file(
    hfss, fieldname, points_filename, solution_parameters, 
    destination_filename, setup, sweep, smooth_field=False, raw_field=True
    ):
    variation_list = []
    for el, value in solution_parameters.items():
        variation_list.append(el + ":=")
        variation_list.append(value)
    points_filename = os.path.join(os.getcwd(), points_filename)
    destination_filename = os.path.join(os.getcwd(), destination_filename)
    solution = f"{setup} : {sweep}"

    hfss.post.ofieldsreporter.CalcStack("clear")
    if raw_field:
        hfss.post.ofieldsreporter.EnterQty(fieldname)
        if smooth_field:
            hfss.post.ofieldsreporter.CalcOp("Smooth")
    else:
        hfss.post.ofieldsreporter.CopyNamedExprToStack(fieldname)
    hfss.post.ofieldsreporter.ExportToFile(
        destination_filename,
        points_filename,
        solution, 
        variation_list,
        [
            "NAME:ExportOption",
            "IncludePtInOutput:="	, True,
            "RefCSName:="		, "Global",
            "PtInSI:="		, True,
            "FieldInRefCS:="	, False
	    ]
    )

# transform into 2 arrays that contain all the frequency dimensions
def run_HFSS(hfss, solution_freqs, export_field_arguments):
    
    solution_parameters = hfss.available_variations.nominal_w_values_dict

    export_field_arguments.update(dict(hfss=hfss, solution_parameters=solution_parameters))


    # get all the fields for each of the available frequencies
    all_solutions = {}
    solved_freqs = []
    for freq in solution_freqs:
        try:
            all_solutions.update({freq: export_HFSS_field(freq, **export_field_arguments)})
            solved_freqs.append(freq)
            print(f"extracted f: {freq}")
        except:
            pass
    solved_freqs = np.array(solved_freqs)

    # create a single np array for each field that contains the information for each frequency
    complex_fields = dict()

    
    for field in export_field_arguments["which_fields"]:
        my_fields = [solution_dict[field] for solution_dict in list(all_solutions.values())]

        complex_fields.update({field: np.stack(my_fields, axis=-1)})
    
    return complex_fields, solved_freqs



def mmstring2float(string_in_mm):
    return float(string_in_mm[:-2])*1e-3


from pyaedt.generic.LoadAEDTFile import load_entire_aedt_file

def custom_basis_frequencies_collector(hfss, setup_name, sweep_name):
        """Get the list of all frequencies which have fields available.
        The project has to be saved and solved in order to see values.
        Returns
        -------
        list of float
            Frequency points.
        """

        my_setup = hfss.setups[0]

        solutions_file = os.path.join(my_setup.p_app.results_directory, "{}.asol".format(my_setup.p_app.design_name))
        fr = []
        if os.path.exists(solutions_file):
            solutions = load_entire_aedt_file(solutions_file)
            for k, v in solutions.items():
                if "SolutionBlock" in k and "SolutionName" in v and v["SolutionName"] == sweep_name and "Fields" in v:
                    try:
                        new_list = [float(i) for i in v["Fields"]["IDDblMap"][1::2]]
                        new_list.sort()
                        fr.append(new_list)
                    except (KeyError, NameError, IndexError):
                        pass
                    
                    return np.array([item for sublist in fr for item in sublist])
        
        
        if fr == []:
            print("couldn't find the basis freqs.. trying with all frequencies")
            # return all frequencies, the basis freqs will be among them
            sol = hfss.post.reports_by_category.standard(setup_name="{} : {}".format(setup_name, sweep_name))
            soldata = sol.get_solution_data()
            if soldata and "Freq" in soldata.intrinsics:
                flat_list = soldata.intrinsics["Freq"]
                units = soldata.units_sweeps["Freq"]
                return (np.array(flat_list)*units_to_value_dict.get(units)).round(0)


        return np.array(flat_list)
    
units_to_value_dict = dict(GHz=1e9, MHz=1e6, kHz=1e3, Hz=1)
  
            





