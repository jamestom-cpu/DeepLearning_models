import numpy as np
import os

from pyaedt.generic.LoadAEDTFile import load_entire_aedt_file

def string_is_numeric(string: str):
    return np.all([char.isdigit() or char in ["+", "-", "e", " ", ".", "\n"] for char in string])



def get_complex_raw_field_values_from_solution_file(myfile, grid_shape = False):
    with open(myfile, "r") as f:
        # discard header line
        f.readline()

        En = []
        points = []
        for line in f.readlines():
            if string_is_numeric(line):
                field_components = line.split("  ")[-1].split(" ")
                field_components.remove("\n")
                En.append(field_components)
                points.append(line.split("  ")[0].split(" "))
        field = np.array((En.remove("\n") if "\n" in En else En), dtype="float64")

        field = field.T[::2]+1j*field.T[1::2]


        points = np.array(points, dtype="float64")

    if grid_shape:
        field = field.reshape(grid_shape)
        X, Y, Z = points.T.reshape(grid_shape)
    return field, [X, Y, Z]


def write_point_file(filename, grid, SI_unit="mm"):
    points = np.stack(grid, axis=0).reshape(3,-1).T
    with open(filename, "w") as f:
        f.write(f"Unit={SI_unit}\n")
        for x, y, z in points:
            f.write(f"{x} {y} {z}\n")


units_to_value_dict = dict(GHz=1e9, MHz=1e6, kHz=1e3, Hz=1)

def custom_basis_frequencies_collector(hfss, setup_name, sweep_name):
        """Get the list of all frequencies which have fields available.
        The project has to be saved and solved in order to see values.
        Returns
        -------
        list of float
            Frequency points.
        """
        
        # in many cases, it is necessary to save the project to have the frequencies available
        hfss.save_project()


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

        print("\nNo solutions available\n")
        return 