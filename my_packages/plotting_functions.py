
from typing import List, Dict
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from matplotlib import cm
from textwrap import fill

from my_packages.model.plotting_funcs_for_morph_model import *
from my_packages.classes.field_classes import *


def plot_field_magnitude_comparison_components(
        field1: Field2D, field2: Field2D, scale="linear", names=[None, None], f_index=0, units="A/m", same_colorbar=False, stack_on_row=True, one_colorbar=True, **kwargs):
    
    params = {
        "figsize": (10, 5) if stack_on_row else (5, 10),
        "title_pad":3,
    }
    
    params = {**params, **kwargs}

    if stack_on_row:
        fig, ax = plt.subplots(2,3, figsize=params["figsize"], constrained_layout=True)
    else:
        fig, ax = plt.subplots(3,2, figsize=params["figsize"], constrained_layout=True)
    name1, name2 = names
    p_objects = {}
    for ii, c in enumerate("xyz"):
        
        # find vmin and vmax
        vmin = min(np.abs(field1[ii]).min(), np.abs(field2[ii]).min()) if same_colorbar else None
        vmax = max(np.abs(field1[ii]).max(), np.abs(field2[ii]).max()) if same_colorbar else None


        kwargs = {
            "field_component": c,
            "scale": scale,
            "f_index": f_index,
            "units": units,
            "vmin": vmin,
            "vmax": vmax
        }

        if stack_on_row:
            plot_objects1 = field1.contour_plot_magnitude(ax = ax[0,ii], title=name1, **{**kwargs, **params})
            plot_objects2 = field2.contour_plot_magnitude(ax = ax[1,ii], title=name2, **{**kwargs, **params})
        else:
            plot_objects1 = field1.contour_plot_magnitude(ax = ax[ii,0], title=name1, **{**kwargs, **params})
            plot_objects2 = field2.contour_plot_magnitude(ax = ax[ii,1], title=name2, **{**kwargs, **params})

        p_objects[f"{c}0"] = plot_objects1 
        p_objects[f"{c}1"] = plot_objects2

        if one_colorbar and same_colorbar:
            plot_objects1["cbar"].remove()

    return fig, ax, p_objects

def plot_field_magnitude_comparison(
        *fields: List[Field2D], component="n", names=[], ax=None,
        suptitle="Field Comparison", scale="linear", f_index=0, units="A/m", 
        same_colorbar=False, keep_only_1_colorbar_and_remove_axes=True):
    
    if ax is None:
        fig, ax = plt.subplots(1,len(fields), figsize=(10,3), constrained_layout=True)

    else:
        ax = np.array(ax).flatten()

    if same_colorbar:
        if component == "n":
            # find vmin and vmax
            vmin = min([f.min for f in fields]) 
            vmax = max([f.max for f in fields]) 
        if component in "xyz":
            c_index = "xyz".index(component)
            vmin = min([np.abs(f[c_index]).min() for f in fields]) 
            vmax = max([np.abs(f[c_index]).max() for f in fields]) 
    else:
        vmin = None
        vmax = None

    kargs = {
        "field_component": component,
        "scale": scale,
        "f_index": f_index,
        "units": units,
        "vmin": vmin,
        "vmax": vmax
    }

    plot_objects =[]
    for ii, f in enumerate(fields):
        plot_objects.append(f.contour_plot_magnitude(title=names[ii] if len(names)>ii else None, ax = ax[ii], **kargs)) 
    
    if same_colorbar and keep_only_1_colorbar_and_remove_axes:
        for ii in range(len(fields)-1):
            plot_objects[ii]["cbar"].remove()
            plot_objects[ii]["ax"].set_ylabel("")



   
    plt.suptitle(suptitle, fontsize=13, fontweight="bold")  
    return plot_objects


# 1d comparison
def compare_fields_on_line(
        field_dictionary: Dict[str, FieldonLine], 
        main_comparison_field: str = None,
        component = "n", units="V/m",
        ax = None, f_index = 0, legend_font_size=10, marker_base_size=5,
        colorlist =None,
        **kwargs
        ):
    if ax is None:
        fig, ax = plt.subplots(figsize=(5,3), constrained_layout=True, dpi=200)
    else:
        fig = ax.figure
    
    markers = list(filter(lambda x: x not in [
        '.',',','1','2','3','4', 1,2,3,4,8,'8', 's', '>', 'v', 'h', 'H', 'p', '*' 
        ], matplotlib.markers.MarkerStyle.markers.keys()))
    
    linestyles = ["dashed", "dotted", "dashdot", "solid"]
    if colorlist is None:
        colorlist="k"
    if len(colorlist)==1:
        colorlist = colorlist*len(field_dictionary)
    elif len(colorlist) < len(field_dictionary):
        colorlist = colorlist + ["k"]*(len(field_dictionary)-len(colorlist))
    else:
        colorlist = colorlist[:len(field_dictionary)]

    for ii, (name, field) in enumerate(field_dictionary.items()):
        if name == main_comparison_field:
            plot_args = {
                "color" : colorlist[ii],
                "alpha" : 0.3,
                "markersize" : marker_base_size + 3,
                "linestyle" : "dashed",
                "linewidth" : 1.5,
                "marker": "s",
                
            }
        else:
            plot_args = {
                "color" : colorlist[ii],
                "alpha" : 1,
                "markersize" : marker_base_size,
                "linestyle" : linestyles[ii],
                "linewidth" : 1,
                "marker": markers[ii]
            }
        field.plot_field(
            component, ax=ax, units=units, f_index=f_index, show_max_values = False,
            label=fill(name, 10, break_long_words=False), **(kwargs | plot_args) 
            )

        # set legend outside of plot and set the label size
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=legend_font_size)
        ax.grid(True, which="both", ls="-", alpha=0.3)
    return fig, ax