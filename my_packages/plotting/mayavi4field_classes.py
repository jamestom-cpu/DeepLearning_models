import mayavi
from mayavi import mlab

import numpy as np
import moviepy
import moviepy.editor as mpy


import attr

from my_packages.plotting import mayavi as mplots
from my_packages.classes.field_classes import Field3D


@attr.s
class QuiverOps:
    figsize = attr.ib()
    arrow_length_factor = attr.ib()
    arrow_thickness = attr.ib()
    units=attr.ib()

class Field3D_Mayavi(Field3D):
    def __init__(
        self, field3d:Field3D, 
        figsize=(800, 500), arrow_length_factor=0.2, arrow_thickness=2, units="A/m", vector_glyph_type="2D") -> None:
        super().__init__(field = field3d.field, freqs=field3d.freqs, grid=field3d.grid) 
        self.QuiverOps = QuiverOps(
            figsize = figsize,
            arrow_length_factor=arrow_length_factor,
            arrow_thickness=arrow_thickness,
            units = units
            )
        self.vector_glyph_type = vector_glyph_type
        
    def start_mayavi_figure(self, magnification=1, figsize=None):
        if figsize is None:
            figsize = self.QuiverOps.figsize
        # start mlab
        mp = mplots.MayaviPlots(size=figsize,vector_glyph_type=self.vector_glyph_type)
        mlab.clf() #clear figure
        mp.magnification=magnification
        self.mayavi_cntr = mp
    
    def plot_at_time(
        self, t, f_index=0, 
        scale_arrows=True, units=None, 
        arrow_length_scale="log", offset=0.1, show_cbar=True, show_axes=True, **kargs
        ):
        
        rfield_class = self.to_time_domain(f_index=f_index, t=t)
        rfield = rfield_class.field[..., 0]
        rfield_abs = np.linalg.norm(rfield, axis=0)
        
        
        if units is None:
            units = self.QuiverOps.units
        
        print(self.QuiverOps.arrow_thickness)

        mp = self.mayavi_cntr
        mp.set_the_scalars_colorbar(rfield, scale_type=arrow_length_scale, offset=offset)
        mp.plot_real_field(rfield, self.grid, scale_arrows, thickness=self.QuiverOps.arrow_thickness, length_factor=self.QuiverOps.arrow_length_factor)       
        
        if show_axes:
            mp.set_axes(mp.quiver)
        if show_cbar:
            mp.build_colorbar(mp.quiver, [rfield_abs.min(), rfield_abs.max()], title=units)

        
    
    def plot_at_phase(self, phase, f_index=0, 
        scale_arrows=True, units=None, 
        arrow_length_scale="log", offset=0.1, show_cbar=True, show_axes=True, **kargs):

        

        # find time instant corresponding to phase
        t = (phase*np.pi/180)/(2*np.pi*self.freqs[f_index])
        self.plot_at_time(t, f_index, scale_arrows, units, arrow_length_scale, offset, show_cbar, show_axes, **kargs)
    
    def update_field_at_time(self, t, f_index=0,
        arrow_length_scale="log", offset=0.1, units=None, color_range = None, show_cbar=True):
        
        if units is None:
            units = self.QuiverOps.units

        """we must already have create a mayavi_cntr class with a scene present and set it as an attribute"""
        mp = self.mayavi_cntr
        rfield_class = self.to_time_domain(f_index=f_index, t=t)
        rfield = rfield_class.field.squeeze()
        if color_range is None:
            rfield_abs = np.linalg.norm(rfield, axis=0)
            color_range = [rfield_abs.min(), rfield_abs.max()]

        # update vec field
        mp.update_sources(rfield, mp.quiver)

        # update_scalars
        scalars = mp.set_the_scalars_colorbar(rfield, reference_field=np.abs(self.field), scale_type=arrow_length_scale, offset=offset)
        mp.update_scalars(scalars, mp.quiver)
        # mp.quiver.mlab_source.set(scalars=offset+np.log10(1+rfield_abs/field_abs.max()).ravel())
    
        if show_cbar:
            # rebuild the colorbar
            mp.build_colorbar(mp.quiver, color_range, title=units)
    
    def run_animation(self, scale_arrows=True, units=None, 
        arrow_length_scale="log", offset=0.1, magnification = 7, figsize=None,
        f_index=0, duration=5, number_of_seconds_per_period=3, show_cbar=True, **kargs):

        self.start_mayavi_figure(magnification, figsize)

        self.plot_at_time(
            0, f_index=f_index,
            scale_arrows=scale_arrows, units=units, 
            arrow_length_scale=arrow_length_scale, 
            offset=offset, show_cbar=show_cbar, **kargs
            )

        # compute the color range
        abs_field = np.linalg.norm(np.abs(self.field[..., f_index]), axis=0)
        color_range = [abs_field.min(), abs_field.max()]

        def update(t):
            #period
            T = 1/self.freqs[f_index]
            # normalize_time
            t=(T/number_of_seconds_per_period)*t

            self.update_field_at_time(
                t, f_index=f_index, arrow_length_scale=arrow_length_scale, 
                color_range=color_range, offset=offset, units=units, show_cbar=show_cbar
                )
            return mlab.screenshot(antialiased=True)
        
        animation = mpy.VideoClip(update, duration=duration)
        return animation
        

        





