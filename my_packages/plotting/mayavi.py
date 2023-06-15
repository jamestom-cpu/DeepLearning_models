import numpy as np
import mayavi
from mayavi import mlab

import numpy as np
import mayavi
from mayavi import mlab

class MayaviSetter():

    def __init__(self, size=(500,250), fgcolor = (0, 0, 0), bgcolor=(1,1,1), vector_glyph_type="2D", **kargs):
        self.figure = mlab.figure(figure="my_figure", size=size, fgcolor=fgcolor, bgcolor=bgcolor, **kargs)
        self.backend = "auto"
        self.NG = False
        self._magnification = 3
        self.vector_glyph_type = vector_glyph_type

    
    @property
    def magnification(self):
        return self._magnification
    
    @magnification.setter
    def magnification(self, value):
        #set magnification
        self.figure.scene.magnification = value
        self._magnification = value


    @property 
    def NG(self):
        return self._NG
    
    @NG.setter
    def NG(self, value: bool):
        mlab.options.offscreen = value
        self._NG = value   

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, bckend):
        mlab.options.backend = bckend
        self._backend = bckend
    
    def set_axes(self, scene, n_labels=4):
        # add axes 
        axs = mlab.axes(scene, xlabel='X', ylabel='Y', zlabel='Z')
        axs.axes.number_of_labels=n_labels
        # add orientation axes
        scene.parent.scene.show_axes=True
    
    def build_colorbar(self, scene, color_range=None, scale="log10", title="values", max_width=100, max_height=500):
        #set options on the mayavi plot
        scene.parent.vector_lut_manager.show_scalar_bar = True
        scene.parent.vector_lut_manager.show_legend = True
        scene.parent.vector_lut_manager.lut.scale=scale
        scene.parent.scalar_lut_manager.lut.scale=scale
        if color_range is not None:
            scene.parent.vector_lut_manager.data_range=color_range

        scene.parent.vector_lut_manager.scalar_bar.unconstrained_font_size = False
        

        scene.parent.vector_lut_manager.scalar_bar.number_of_labels = 15
        scene.parent.vector_lut_manager.use_default_name = False
        scene.parent.vector_lut_manager.scalar_bar.title = title
        scene.parent.vector_lut_manager.title_text_property.font_size = 21

        scene.parent.vector_lut_manager.label_text_property.italic = False
        scene.parent.vector_lut_manager.label_text_property.font_family = 'times'
        scene.parent.vector_lut_manager.label_text_property.shadow=False 
        scene.parent.vector_lut_manager.label_text_property.font_size = 16
        
        

        scene.parent.vector_lut_manager.title_text_property.italic = False
        scene.parent.vector_lut_manager.title_text_property.font_family = "times"
        scene.parent.vector_lut_manager.title_text_property.shadow = True

        # set size parameters
        scene.parent.vector_lut_manager.scalar_bar.bar_ratio = 0.3
        scene.parent.vector_lut_manager.scalar_bar.maximum_width_in_pixels = max_width
        scene.parent.vector_lut_manager.scalar_bar.maximum_height_in_pixels = max_height
        scene.parent.vector_lut_manager.scalar_bar.text_pad = 2
        scene.parent.vector_lut_manager.scalar_bar.title_ratio = 0.25
    
    def set_vector_glyph_type(self, scene, type:"str"):
        if type == "2D":
            scene.glyph.glyph_source.glyph_source = scene.glyph.glyph_source.glyph_dict['glyph_source2d']
            scene.glyph.glyph_source.glyph_source.glyph_type = 'arrow'
        
        if type == "3D":
            scene.glyph.glyph_source.glyph_source = scene.glyph.glyph_source.glyph_dict["arrow_source"]

            
class MayaviPlots(MayaviSetter):
    def init(self, size=(500,250), fgcolor = (0, 0, 0), bgcolor=(1,1,1), vector_glyph_type="2D", **kargs):
        super().__init__(size, fgcolor, bgcolor, vector_glyph_type, **kargs)

    def set_the_scalars_colorbar(self, rfield, reference_field=None, scale_type="log", offset=0.1):
        
        field_abs = np.linalg.norm(rfield, axis=0)
        if reference_field is None:
            reference_field = field_abs
        ref_abs = np.linalg.norm(reference_field, axis=0)

        if scale_type == "linear":
            self.scalars = offset+field_abs/ref_abs.max().ravel()
        elif scale_type == "log":
            self.scalars = offset+np.log10(1+field_abs/ref_abs.max()).ravel()
                
        return self.scalars

    
    def plot_real_field(
        self, rfield: np.array, grid: np.array, scale_arrows=True, use_default_scalars=False,
        thickness = 2, length_factor=0.2
        ):
        """the grid should be passed in [mm]"""
        grid = grid*1e3
        x, y, z = grid
        u, v, w = rfield
        field_abs = np.linalg.norm(rfield, axis=0)

        print(thickness)

        if use_default_scalars:
            self.set_the_scalars_colorbar(rfield)
            
        # plot the quiverplot
        if not scale_arrows:
            self.quiver = mlab.quiver3d(x,y,z,u,v,w, line_width=thickness, scale_factor=length_factor, scale_mode="none")
        else:
            self.quiver = mlab.quiver3d(x,y,z,u,v,w, line_width=thickness, scale_factor=length_factor, scale_mode="scalar", scalars=self.scalars)
        
        self.change_glyph_type(self.vector_glyph_type)

    def change_glyph_type(self, glyph_type):
        self.set_vector_glyph_type(self.quiver, glyph_type)

    def update_sources(self, rfield, scene):
        u,v,w = rfield
        # set the sources
        scene.mlab_source.set(u=u, v=v, w=w)
    
    def update_scalars(self, scalars, scene):
        scene.mlab_source.set(scalars=scalars)

    def show_screenshot(self, scene=None):
        mlab.screenshot()
        return mlab.screenshot(scene)
