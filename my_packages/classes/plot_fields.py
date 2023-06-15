import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import math as m
import seaborn as sns
from functools import partial
import copy
from .aux_classes import Grid
import decimal

class Field2D_plots():
    def plot_XY_slice(self, domain_index=0, func_on_field=np.squeeze, axes=None, backend=None):
        if backend is None:
            pass
        elif backend == "inline":
            try:
                import IPython
                shell = IPython.get_ipython()
                shell.enable_matplotlib(gui = 'inline')
            except:
                pass
        else:
            plt.close("all")
            plt.switch_backend(backend)
            plt.ion()

        if axes is None: 
            fig, axes = plt.subplots(1,2, figsize=(12,5))
        
        # chose the frequency
        field2D = self.field[...,domain_index]

        return plot_field_and_mag(field=func_on_field(np.expand_dims(field2D, axis=-2)), r=np.expand_dims(self.grid, axis=-1), axes=axes, z_level=0)
    
    def contour_plot_all_components_magnitude(self, f_index=0, ax=None, **kargs):

        if ax is None:
            _, ax = plt.subplots(2,2, figsize=(12,8), constrained_layout=True)

        for ii, comp in enumerate(["x", "y", "z", "norm"]):
            self.contour_plot_magnitude(
                comp, ax=ax[ii%2, ii//2], f_index = f_index, **kargs
                )
            ax[ii%2, ii//2].set_title(f"Magnitude of the Field Norm" if comp=="norm" else f"{comp} Component")


    def contour_plot_magnitude(
            self, field_component="n", f_index=0, scale="linear", units="A/m", ax=None, 
            title=None,  keep_linear_scale_on_colormap=False, vmin=None, vmax=None, title_pad=0,
            *args, **kargs
            ):

        Ex, Ey, Ez = self.field[..., f_index]

        if field_component=="x":
            plt_field = np.abs(Ex)
            def_title = "X Component Magnitude"
        elif field_component=="y":
            plt_field = np.abs(Ey)
            def_title = "Y Component Magnitude"
        elif field_component == "z":
            plt_field = np.abs(Ez)
            def_title = "Z Component Magnitude"
        elif field_component in ["n", "norm", None]:
            plt_field = self.norm[..., f_index]
            def_title = "Vector Complex Magnitude"
        else:
            raise KeyError(f"the field component cannot be set to `{field_component}`")
        
        if title is None:
            title = def_title
        else:
            title = f"{title} - {def_title}"

        if ax is None:
            fig, ax = plt.subplots(figsize=(10,6))

        if self.axis == "x":
            _,X,Y= self.grid
            labels = ["Y", "Z"]
        if self.axis == "y":
            X,_,Z= self.grid
            labels = ["X", "Z"]
        if self.axis == "z":
            X,Y,_ = self.grid
            labels = ["X", "Y"]

        locator = ticker.LogLocator() if scale == "log" else ticker.LinearLocator()
        
        plot_kargs = dict(
            n_levels = 50, 
            cmap = cm.jet,
        )

        plot_kargs.update(kargs)
        n_levels = plot_kargs["n_levels"]
        del plot_kargs["n_levels"]

        if vmin is None:
            vmin = plt_field.min()
        
        if vmax is None:
            vmax = plt_field.max()

        cs_range_log = np.logspace(np.log10(vmin+np.finfo(np.float64).eps),np.log10(vmax+2*np.finfo(np.float64).eps), n_levels)
        cs_range_lin = np.linspace(vmin+np.finfo(np.float64).eps, vmax+2*np.finfo(np.float64).eps, n_levels)

        
        if np.ndim(X) == 3:
            X = X.squeeze()
        if np.ndim(Y) == 3:
            Y = Y.squeeze()

        cs = ax.contourf(X, Y, plt_field, 
                        levels = cs_range_log if scale=="log" else cs_range_lin, 
                        locator=locator, 
                        **plot_kargs
                        )


        if scale=="log" and not keep_linear_scale_on_colormap:
            ticks = np.geomspace(vmin+np.finfo(np.float64).eps, vmax+np.finfo(np.float64).eps, 11)
        else:
            ticks = ticker.LinearLocator().tick_values(vmin, vmax)
        
        # set the max number of decimals to 2
        
        def find_decimal_order(val: float):
            return -int(decimal.Decimal(str(val)).as_tuple().exponent)
        
        min_decimal_order = find_decimal_order(min(ticks))
        if min_decimal_order<3:
            ticks = np.round(ticks, min_decimal_order+2)
        else:
            ticks = np.round(ticks, min_decimal_order+1)
        
    
        cbar = plt.colorbar(cs, ticks=ticks)
        cbar.ax.set_ylabel(units, rotation=270, labelpad=15)

        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{1000*x:.1f}"))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{1000*x:.1f}"))

        ax.set_xlabel(f"{labels[0]} [mm]")
        ax.set_ylabel(f"{labels[1]} [mm]")
        ax.set_title(title, pad=title_pad)

        return dict(ax=ax, cs=cs, cbar=cbar)
    
    def contour_plot_phase(self, field_component="n", f_index=0, units="A/m", ax=None, title=None,  keep_linear_scale_on_colormap=False, vmin=None, vmax=None):

        Ex, Ey, Ez = self.field[..., f_index]

        if field_component=="x":
            plt_field = np.angle(Ex, deg=True)
            def_title = "X Component Phase"
        elif field_component=="y":
            plt_field = np.angle(Ey, deg=True)
            def_title = "Y Component Phase"
        elif field_component == "z":
            plt_field = np.angle(Ez, deg=True)
            def_title = "Z Component Phase"
        elif field_component in ["n", "norm", None]:
            plt_field = np.angle(self.complex_norm[..., f_index], deg=True)
            def_title = "Vector Complex Phase"
        else:
            raise KeyError(f"the field component cannot be set to `{field_component}`")
        
        if title is None:
            title = def_title
        else:
            title = f"{title} - {def_title}"

        if ax is None:
            fig, ax = plt.subplots(figsize=(10,6))

        if self.axis == "x":
            _,X,Y= self.grid
            labels = ["Y", "Z"]
        if self.axis == "y":
            X,_,Z= self.grid
            labels = ["X", "Z"]
        if self.axis == "z":
            X,Y,_ = self.grid
            labels = ["X", "Y"]
        
        if vmin is None:
            vmin = plt_field.min()
        
        if vmax is None:
            vmax = plt_field.max()

        locator = ticker.LinearLocator()

        n_levels = 50

        cs_range_lin = np.linspace(vmin+np.finfo(np.float64).eps,vmax+2*np.finfo(np.float64).eps, n_levels)




        cs = ax.contourf(X*1e3, Y*1e3, plt_field, 
                        levels = cs_range_lin, 
                        locator=locator, 
                        cmap=cm.jet
                        )


        ticks = ticker.LinearLocator().tick_values(vmin, vmax)
    
        cbar = plt.colorbar(cs, ticks=ticks)
        cbar.ax.set_ylabel(units, rotation=270, labelpad=15)

        ax.set_xlabel(f"{labels[0]} [mm]")
        ax.set_ylabel(f"{labels[1]} [mm]")
        ax.set_title(title)

        return dict(ax=ax, cs=cs, cbar=cbar)


class Field3D_plots():
# add plotting functions to class
    def plot_at_phase(self, phase, f_index=0, units="A/m", arrow_color=None, ax=None):
        f = self.freqs[f_index]
        phase_rad = (np.pi/180)*phase
        time_point = phase_rad/2*np.pi*f

        _, x, y, z = self.grid.shape

        x_down = x//7; y_down = y//7; z_down = z//3

        downsample_step={"X":x_down, "Y":y_down, "Z":z_down}

        quiver, surfaces, scam = self.to_time_domain(f_index=f_index, t=time_point).plot_3D(   
            percentile_clip=100, cmap="jet", 
            downsample_step=downsample_step,
            scale_color_arrows=False, arrow_color=arrow_color, log_colorbar=True, 
            units=units, change_to_mm=True, title=f"{int(phase)}deg Phase", ax=ax
        )
        return quiver, surfaces, scam


    def plot_XY_slice(self, z_level, domain_index=0, func_on_field=np.squeeze, axes=None, backend=None):
        if backend is None:
            pass
        elif backend == "inline":
            try:
                import IPython
                shell = IPython.get_ipython()
                shell.enable_matplotlib(gui = 'inline')
            except:
                pass
        else:
            plt.close("all")
            plt.switch_backend(backend)
            plt.ion()

        if axes is None: 
            fig, axes = plt.subplots(1,2, figsize=(12,5))

        field3D = self.field[..., domain_index]

        return plot_field_and_mag(field=func_on_field(field3D), r=self.grid, axes=axes, z_level=z_level)
    
    def plot_3D(self, func_on_field = lambda x: x, domain_index=0, orientation_dict=dict(xy=2, xz=1, yz=1), ax=None, 
    downsample_step = {"X":3, "Y":3, "Z":5}, percentile_clip=90, 
    cmap="jet", scale_color_arrows=True, arrow_color=None, log_colorbar = False, units="units_keyword", change_to_mm=False, title=None):
        
        # if backend is None:
        #     pass
        # elif backend == "inline":
        #     try:
        #         import IPython
        #         shell = IPython.get_ipython()
        #         shell.enable_matplotlib(gui = 'inline')
        #     except:
        #         pass
        # else:
        #     plt.close("all")
        #     plt.switch_backend(backend)
        #     plt.ion()
        if ax is None:
            _, ax = plt.subplots(figsize=(12,5), subplot_kw={'projection': '3d'})
            ax.set_title(title)
        
        # chose the frequency
        field3D = self.field[...,domain_index]

        ###############################
        quiver, surfaces, scam = plot_3Dfield(
            func_on_field(field3D), self.grid, orientation_dict, ax=ax, 
            downsample_step = downsample_step, percentile_clip=percentile_clip, 
            cmap=cmap, scale_color_arrows=scale_color_arrows, arrow_color=arrow_color, log_colorbar = log_colorbar, units=units, change_to_mm=change_to_mm)
        return quiver, surfaces, scam 


def phasor_to_real_value(complex_field, f, t=0):
    # f(t) = Re{Fexp(jwt)}
    field = np.real(np.multiply(np.exp(1j*2*m.pi*f*np.array(t)),np.expand_dims(complex_field, axis=0).T)).T
    return field  

def get_linewidth(field_norm, normalizing_lw = None, lw_factor = 5):

    if normalizing_lw is None:
        normalizing_lw = field_norm.max()
    
    return lw_factor*field_norm/normalizing_lw


def plot_field_and_mag(field, r, axes, z_level=0.3, domain_index=0):

    
    r = np.array(r).swapaxes(1,2)
    field = field.swapaxes(1,2)

    ax1, ax2 = axes

    X,Y,Z = r
    z_index=np.argmin(np.abs(Z-z_level))

    X=X[0,:,z_index]; Y=Y[:,0,z_index]
    field_slice = field[:,:,:,z_index]

    field_norm = np.sqrt(np.einsum("i...,i...", field_slice, field_slice))

    lw = get_linewidth(field_norm, normalizing_lw=None, lw_factor=5)

    ax1.streamplot(X,Y, field_slice[0], field_slice[1], color=field_norm, cmap="Blues", linewidth=lw)
    ax1.set_xlabel("X[m]")
    ax1.set_ylabel("Y[m]")

    field_norm_table = pd.DataFrame(field_norm, dtype="float64")
    my_X = np.round(pd.Index(X, name="X"), 2)
    my_Y = np.round(pd.Index(Y, name="Y"), 2)       

    my_tb = field_norm_table.set_axis(my_X, axis="columns").set_axis(my_Y, axis="index").iloc[::-1,:]
    sns.heatmap(my_tb, ax=ax2, cbar_kws={"label":"V/m"})
    ax2.set_xlabel("X[m]")
    ax2.set_ylabel("Y[m]")

    ax1.set_aspect("equal")
    ax1.margins(0,0)
    # ax2.set_aspect("equal")
    ax2.margins(0,0)




def plot_3Dfield(
    field3d, r, orientation_dict, ax=None, 
    downsample_step = {"X":3, "Y":3, "Z":5}, percentile_clip=90, 
    cmap="jet", scale_color_arrows=True, arrow_color=None, log_colorbar = False, units="units_keyword", change_to_mm = False
    ):

    r = np.array(r).swapaxes(1,2)

    if change_to_mm:
        r=r*1e3

    field3d = field3d.squeeze().swapaxes(1,2)

    # create surfaces
    X,Y,Z = r


    # get the field intensity 3D
    field_intensity = np.linalg.norm(field3d, axis=0)

    cm = getattr(plt.cm, cmap)

    if ax is None:
        _, ax = plt.subplots(figsize=(12,12), subplot_kw={'projection': '3d'})
    

    # normalizing_values_planes = np.concatenate(
    #     [np.stack(list(get_field_intensity_on_slice(X,Y,Z,field_intensity, orientation=key, n_surfaces=value).values()), axis=0) 
    #     for key, value in surface_dict.items()], axis=0)

    normalizing_values_whole_field = field_intensity
    
    vmin = normalizing_values_whole_field.min()
    # vmax = normalizing_values_whole_field.max()

    # for k, value in orientation_dict.items():
    #     simple_plot_3D_surface(field3d, r, n_surfaces=value, ax=ax, direction=k)

    # boundary norm
    bounds_log = np.geomspace(vmin+np.finfo(np.float64).eps, np.percentile(normalizing_values_whole_field+2*np.finfo(np.float64).eps, percentile_clip), 30)
    bounds_linear = np.linspace(vmin+np.finfo(np.float64).eps, np.percentile(normalizing_values_whole_field+2*np.finfo(np.float64).eps, percentile_clip), 30)
    # boundary_norm = plt.cm.colors.BoundaryNorm(boundaries=bounds, ncolors=256, extend='max')

    # fix some parameters for the boundary norm function. 
    boundary_normalization = partial(plt.cm.colors.BoundaryNorm, ncolors=256, extend="max")

    my_norm = (boundary_normalization(boundaries=bounds_log) if log_colorbar else boundary_normalization(boundaries=bounds_linear))

    # Create the 4th color-rendered dimension
    scam = plt.cm.ScalarMappable(
        norm=my_norm,
        cmap=cm # see https://matplotlib.org/examples/color/colormaps_reference.html
    )

    for orientation, number_of_planes in orientation_dict.items():
        my_planes = get_field_intensity_on_slice(X,Y,Z,field_intensity, orientation, number_of_planes)
        surfaces = []
        for shift, value in my_planes.items():
            print(value.shape)
            color_surface = scam.to_rgba(value)
            surface = simple_plot_3D_surface(X,Y,Z, orientation, shift, ax=ax, scalar_mappable = color_surface, alpha=0.9)
            # q.set_clim(vmin,vmax)
            surfaces.append(surface)
    arrow_scam = scam if scale_color_arrows else None
    # print(arrow_scam)
    quiver = simple_quiverplot_3D(field3d, r, downsample_step=downsample_step, ax = ax, scam= arrow_scam, arrow_color=arrow_color)
    
    plt.colorbar(scam, ax=ax, orientation='vertical', label=units, extend="max")

    if change_to_mm:
        ax.set_xlabel("X[mm]")
        ax.set_ylabel("Y[mm]")
        ax.set_zlabel("Z[mm]")
    else:
        ax.set_xlabel("X[m]")
        ax.set_ylabel("Y[m]")
        ax.set_zlabel("Z[m]")

    return quiver, surfaces, scam
    


def simple_plot_3D_surface(X,Y,Z, orientation, shift, scalar_mappable=None, ax=None, **kargs):

    if orientation == "xy":           
        X2d = X[:,:,0]
        Y2d = Y[:,:,0]
        q = ax.plot_surface( X2d, Y2d, np.ones(X2d.shape)*shift, facecolors=scalar_mappable, **kargs)

    if orientation == "xz":        
        X2d = X[0,:,:]
        Z2d = Z[0,:,:]
        q = ax.plot_surface(np.ones(X2d.shape)*shift, X2d, Z2d, facecolors=scalar_mappable,  **kargs)

    if orientation == "yz":
                
        Y2d = Y[:,0,:]
        Z2d = Z[:,0,:]
        q = ax.plot_surface(Y2d, np.ones(Y2d.shape)*shift, Z2d, facecolors=scalar_mappable, **kargs)

    return q


def get_field_intensity_on_slice(X,Y,Z, field_intensity, orientation="xy", n_surfaces=1):
    if orientation == "xy":
        # surface plot positions
        if n_surfaces > 1:
            positions = np.round(np.linspace(Z[0,0,:].min(), Z[0,0,:].mean(), n_surfaces), 2)
        else:
            positions = [np.round(Z[0,0,-1], 2)]

        #take the field intensity along surface
        # print([np.argmin(np.abs(Z[0,0,:]-plane)) for plane in positions])
        fi_list = {plane: field_intensity[:, : , np.argmin(np.abs(Z[0,0,:]-plane))] for plane in positions}
        return fi_list

    if orientation == "xz":
        # surface plot positions
        if n_surfaces > 1:
            positions = np.round(np.linspace(Y[0,:,0].min(), Y[0,:,0].mean(), n_surfaces), 2)
        else: 
            positions = [np.round(Y[0,0,0], 2)]

        #take the field intensity along surface
        # print([np.argmin(np.abs(Y[:,0,0]-plane)) for plane in positions])
        fi_list = {plane: field_intensity[np.argmin(np.abs(Y[:,0,0]-plane)), : , :] for plane in positions}
        return fi_list


    if orientation == "yz":
        # surface plot positions
        if n_surfaces > 1:
            positions = np.round(np.linspace(X[0,:,0].mean(), X[0,:,0].max(), n_surfaces), 2)
        else: 
            positions = [np.round(X[0,-1,0], 2)]
        #take the field intensity along surface
   
        # print([np.argmin(np.abs(X[0,:,0]-plane)) for plane in positions])
        fi_list = {plane: field_intensity[:, np.argmin(np.abs(X[0,:,0]-plane)) , :] for plane in positions}
        return fi_list


def simple_quiverplot_3D(field3d, r, downsample_step, ax, scam=None, arrow_color="black"):

    def downsample(field3d, downsample_step):
        str2axis=dict(X=1, Y=0, Z=2)

        downsampled_field = field3d.copy()

        for axis, value in downsample_step.items():
            axis = str2axis[axis]
            # print(axis, value)
            downsampled_field = np.take(downsampled_field, indices=np.arange(1,field3d.shape[axis+1], value), axis=(axis+1))
            # print(downsampled_field.shape)
        return downsampled_field
 


    downsampled_field = downsample(field3d, downsample_step) 
    downsampled_r = downsample(np.array(r), downsample_step)


    
    field_intensity = np.linalg.norm(downsampled_field, axis=0)

    # # get negative values for sinks and positive value for source
    # dot_p = np.tensordot(p, downsampled_field, axes=1)
    # sign = np.ones(dot_p.shape); sign[dot_p<0] = -1
    # signed = field_intensity*sign


    Xfield,Yfield,Zfield = downsampled_field
    X,Y,Z = downsampled_r

    xscale = X.ptp()
    yscale = Y.ptp()
    zscale = Z.ptp()

    # scale = np.linalg.norm(np.array([xscale, yscale, zscale]))

    print(field_intensity.shape)
    c = scam.to_rgba(field_intensity.ravel()) if scam else arrow_color
    

    q = ax.quiver(
        X,Y,Z, Xfield,Yfield,Zfield,
        pivot='middle', length=0.2, normalize=True, lw=1.3, alpha=0.8
    )

    # q.set_edgecolor(c)
    q.set_color(c) 

    return q

def plot_2d_field_pcolor(
        field: np.ndarray, ax=None,  grid: Grid = None, title: str = None,
        include_colorbar = True, units="V/m",
        **kargs):
    """Plot a 2D field using pcolormesh"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(10,3), dpi=100)
    else:
        fig = ax.get_figure()

    axes_names = [axis for axis in "xyz" if len(getattr(grid, axis)) != 1]
    axes = [getattr(grid, axis)*1e3 for axis in axes_names]

    
    # plot the field 
    png = ax.pcolormesh(*axes, field.T, **kargs)

    # set the axis labels
    ax.set_xlabel(f"{axes_names[0]} [mm]")
    ax.set_ylabel(f"{axes_names[1]} [mm]")
    if title is not None:
        ax.set_title(title)

    # set the colorbar
    if include_colorbar:
        fig.colorbar(png, ax=ax, label=units)

    return png