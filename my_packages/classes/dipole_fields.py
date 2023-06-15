import numpy as np
import json
from types import SimpleNamespace
import copy


from mpl_toolkits.mplot3d import Axes3D
from my_packages import spherical_coordinates, partial_image_expansion
from my_packages.classes import field_classes
from my_packages.classes.field_classes import Field3D
from my_packages.classes import dipole_array as d_array
from . import green_sols
from .model_components import UniformEMSpace, Substrate


from my_packages import spherical_coordinates, field_calculators
from my_packages import persist_save

    
    
##########################################################
from .dipole_field_base import Dipole_Field, Dipole_Field_Base
#########################################################

class MyJsonEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, DFHandler_over_Substrate):
                return persist_save.convert_dict_to_numpy(persist_save.Model2JSON(obj).__dict__)
            return super().default(obj)


class InfElectricDipole(Dipole_Field):
    def H_evaluation(method):
        def wrapper(self: 'InfElectricDipole', *args, **kwargs):
            G = method(self,*args, **kwargs)
            sources = self.dipole_array.dipole_moments

            H = np.einsum("i...k, i...k -> ...k", G, sources)      
            self.H = field_classes.Field3D(H, self.f, self.r)
            return self.H
        return wrapper
    
    def E_evaluation(method):
        def wrapper(self: 'InfElectricDipole', *args, **kwargs):
            G = method(self,*args, **kwargs)
            sources = self.dipole_array.dipole_moments

            E = np.einsum("i...k, i...k -> ...k", G, sources)      
            self.E = field_classes.Field3D(E, self.f, self.r)
            return self.E
        return wrapper
    
    def __init__(self, EM_space: UniformEMSpace, dipole_array=None):
        super().__init__(EM_space, dipole_array)
        self.dipole_array.set_dipole_type("Electric")
    
    @property
    def green_solutions(self):
        return green_sols.GreenSolutions_ElectricSubstr(self)
    
    def add_substrate(self, substrate: partial_image_expansion.Substrate):
        return InfElectricDipole_with_Substrate(substrate, self.eps_r, self.mu_r, self.sigma, self.r, self.dipole_array)

    def keep_valid_dipoles(self, GND_h=0)-> np.ndarray:
        positions = self.dipole_array.r0
        invalid_args = np.where(positions[-1] < GND_h, positions)
        return invalid_args

    def get_image_positions(self, GND_h=0):
        dipole_positions = copy.deepcopy(self.dipole_array.r0).astype("float64")
        dipole_positions[:, -1] = 2*GND_h-dipole_positions[:, -1]
        return dipole_positions
    
    def get_image_orientations(self):
        """
        For magnetic dipole sources the image dipoles of:

        - Horizontal Dipoles => Oriented In the Opposite Direction
        - Vertical Dipoles => Oriented in Same Direction
        
        """
        orientations = self.dipole_array.orientations
        image_orientation = np.stack([orientations[:, 0], np.pi+orientations[:, 1]], axis=1)
        return image_orientation
    
    def get_image_sources(self, GND_h):
        
        freqs = []
        moments = []
        for dipole in self.dipole_array.dipoles:
            r0_x, r0_y, r0_z = dipole.r0
            if r0_z >= GND_h:
                moments.append(dipole.dipole_moment.signal)
        
        return np.stack(moments, axis=0)
    
    def get_image_array(self, GND_h = 0):
        positions = self.get_image_positions(GND_h)
        orientations = self.get_image_orientations()
        moments = self.get_image_sources(GND_h)

        image_dipole_ar = d_array.DipoleSourceArray(self.f, positions, orientations=orientations, moments=moments)
        
        return image_dipole_ar
    
    def add_image_sources(self, GND_h=0):
        image_array = self.get_image_array(GND_h)
        total_array = d_array.combine_arrays(self.dipole_array, image_array)

        return InfElectricDipole(EM_space=self.EM_space, dipole_array=total_array)
    
    def from_G_to_evaluation(method):
        def wrapper(self: 'InfElectricDipole', *args, **kwargs):
            G = method(self,*args, **kwargs)
            sources = self.dipole_array.dipole_moments

            E = np.einsum("i...k, i...k -> ...k", G, sources)      
            self.E = field_classes.Field3D(E, self.f, self.r)
            return self.E
        return wrapper

    @E_evaluation
    def get_E_uniform_space(self) -> field_classes.Field3D:
        return self.green_solutions.Gee_uniform_space()
    
    @E_evaluation
    def get_E_image_method(self, GND_h=0) -> field_classes.Field3D:
        return self.green_solutions.Gee_image_method(GND_h)


    # aliases
    # backward compatibility
    @property
    def evaluate_electric_field_image_method(self):
        return self.get_E_image_method
    
    @property
    def evaluate_electric_field(self):
        return self.get_E_uniform_space
    
    
    



class InfElectricDipole_with_Substrate(InfElectricDipole):
    def __init__(
        self, substrate: partial_image_expansion.Substrate | None, 
        dipole_array: d_array.DipoleSourceArray,
        EM_space: UniformEMSpace
        ):
        super().__init__(EM_space, dipole_array)
        self.substrate = substrate
        
    
    @property
    def green_solutions(self):
        return green_sols.GreenSolutions_ElectricSubstr(self)
    
    def H_evaluation(method):
        def wrapper(self: 'InfElectricDipole_with_Substrate', *args, **kwargs):
            G = method(self,*args, **kwargs)
            sources = self.dipole_array.dipole_moments

            H = np.einsum("i...k, i...k -> ...k", G, sources)      
            self.H = field_classes.Field3D(H, self.f, self.r)
            return self.H
        return wrapper
    
    def E_evaluation(method):
        def wrapper(self: 'InfElectricDipole_with_Substrate', *args, **kwargs):
            G = method(self,*args, **kwargs)
            sources = self.dipole_array.dipole_moments

            E = np.einsum("i...k, i...k -> ...k", G, sources)      
            self.E = field_classes.Field3D(E, self.f, self.r)
            return self.E
        return wrapper
    
    def add_partial_image_sources(self, GND_h, N=15):
        partial_image_class = partial_image_expansion.PartialImageExpansion(
            dipole_ar=self.dipole_array, substrate=self.substrate, medium_epsr=self.eps_r, GND_h=GND_h
            )
        image_array = partial_image_class.get_image_arrays(N)

        all_sources = d_array.combine_arrays(self.dipole_array, image_array)
        return InfElectricDipole_with_Substrate(eps_r=self.eps_r, mu_r = self.mu_r, sigma=self.sigma, r=self.r, dipole_array=all_sources, substrate=self.substrate)
    
    

    @E_evaluation
    def get_E_partial_image_expansion(self, GND_h=None, N=15) -> field_classes.Field3D:
        # use image theory to take the presence of the ground plane into account  
        if GND_h is None:
            GND_h = -self.substrate.thickness
        return self.green_solutions.Gee_partial_image_expansion(GND_h = GND_h, N=N)  
        
    
    @H_evaluation
    def get_H_deconstructed_interpolation(self, GND_h=None, N=15) -> field_classes.Field3D:
        if GND_h is None:
            GND_h = -self.substrate.thickness
        return self.green_solutions.Geh(GND_h=GND_h, N=N)

    

    @H_evaluation
    def get_H_uniform_space(self) -> field_classes.Field3D:
        return self.green_solutions.Geh_uniform_space()
    
    

    @H_evaluation
    def get_H_image_method(self, GND_h= None) -> field_classes.Field3D:
        if GND_h is None:
            GND_h = -self.substrate.thickness
        return self.green_solutions.Geh_image_method(GND_h)
    

    @H_evaluation
    def get_H_partial_image_expansion_epsr(self, GND_h= None, N=15) -> field_classes.Field3D:
        if GND_h is None:
            GND_h = -self.substrate.thickness
        return self.green_solutions.Geh_partial_image_expansion(GND_h, N)
    
    
    @H_evaluation
    def evaluate_magnetic_field_over_PEC_interpolated(self, GND_h= None, xi=0.5, N=15) -> field_classes.Field3D:
        if GND_h is None:
            GND_h = -self.substrate.thickness
        return self.green_solutions.Geh_directly_interpolated_btw_image_methods(GND_h, xi, N)
    
    def __evaluate_magnetic_field_with_curl(self):
        assert self.E is not None, "you must first calculate the magnetic field"
        self.H = -self.E.curl()/(1j*self.f*2*np.pi*self.mu0*self.mu_r)
        return self.H
    # aliases
    # ease of use
    def evaluate_E(self, *args, method="partial_image_expansion", **kargs)-> Field3D:
        if method in ["partial_image_expansion", "partial image expansion", "pie"]:
            return self.get_E_partial_image_expansion(*args, **kargs)
        elif method in ["image_method", "image method"]:
            return self.get_E_image_method(*args, **kargs)
        elif method in ["uniform_space", "uniform space"]:
            return self.get_E_uniform_space(*args, **kargs)
    
    def evaluate_H(self, *args, method="deconstructed_interpolation", **kargs)-> Field3D:
        if method in ["deconstructed_interpolation", "deconstructed interpolation", "di"]:
            return self.get_H_deconstructed_interpolation(*args, **kargs)
        elif method in ["image_method", "image method", "im"]:
            return self.get_H_image_method(*args, **kargs)
        elif method in ["uniform_space", "uniform space", "us"]:
            return self.get_H_uniform_space(*args, **kargs)
        elif method in ["partial_image_expansion_epsr", "partial image expansion epsr", "pie_epsr"]:
            return self.get_H_partial_image_expansion_epsr(*args, **kargs)
        

    def get_handler_for_single_dipole(self, orientation: str, *args, **kargs)->"InfElectricDipole_with_Substrate":
        x, y = self.EM_space.r.x, self.EM_space.r.y

        x0 = (x[0]+x[-1])/2; y0 = (y[0]+y[-1])/2
        r0 = np.array([x0, y0])

        my_kwargs = {"r0": r0}
        my_kwargs.update(kargs)

        single_dipole_array = self.dipole_array.get_single_test_dipole_on_array(orientation, type="Electric", *args, **my_kwargs)
        return InfElectricDipole_with_Substrate(dipole_array=single_dipole_array, substrate=self.substrate, EM_space=self.EM_space)

    
    # backwards compatibility
    @property
    def evaluate_electric_field_over_PEC(self):
        return self.get_E_partial_image_expansion

    @property
    def evaluate_magnetic_field_over_PEC_partial_im_expansion(self):
        return self.get_H_partial_image_expansion_epsr

    @property
    def evaluate_magnetic_field_over_PEC_images(self):
        return self.get_H_image_method
    
    @property
    def evaluate_magnetic_field(self):
        return self.get_H_uniform_space
    
    @property
    def evaluate_magnetic_field_over_PEC(self):
        return self.get_H_deconstructed_interpolation





class InfCurrentLoop(Dipole_Field):

    def __init__(self, EM_space: UniformEMSpace, dipole_array=None,
                 magnetic_current_source=False
                ):
        super().__init__(EM_space, dipole_array)
        self.dipole_array.set_dipole_type("Magnetic")

        self._magnetic_current_source = False
        self.magnetic_current_source = magnetic_current_source

    @property
    def green_solutions(self):
        return green_sols.GreenSolutions_MagneticSubstr(self) 
    
    @property
    def magnetic_current_source(self):
        return self._magnetic_current_source
    
    @magnetic_current_source.setter
    def magnetic_current_source(self, magnetic_current: bool):
        
        if (self.magnetic_current_source - magnetic_current)==0:
            return
        if self.magnetic_current_source and not magnetic_current:
            self.dipole_array.dipole_moments = self.dipole_array.dipole_moments*(1j*2*np.pi*self.f*self.mu0*self.mu_r)[None, :]
            self._magnetic_current_source = magnetic_current
            return
        if not self.magnetic_current_source and magnetic_current:
            self.dipole_array.dipole_moments = self.dipole_array.dipole_moments/(1j*2*np.pi*self.f*self.mu0*self.mu_r)[None, :]
            self._magnetic_current_source = magnetic_current
            return


    
    def keep_valid_dipoles(self, GND_h=0):
        positions = self.dipole_array.r0

        valid_indices = self.dipole_array.r0[:,-1]>=GND_h

        valid_positions = self.dipole_array.r0[valid_indices]
        valid_orientations = self.dipole_array.orientations[valid_indices]
        if self.dipole_array.dipole_moments is None:
            valid_moments = None
        else:
            valid_moments = self.dipole_array.dipole_moments[valid_indices]

        valid_array = d_array.DipoleSourceArray(self.f, valid_positions, valid_orientations, valid_moments)
        self.dipole_array = valid_array

        return self


    def get_image_orientations(self):
        """
        For magnetic dipole sources the image dipoles of:

        - Horizontal Dipoles => Oriented In the Same Direction
        - Vertical Dipoles => Oriented in Opposite Direction
        
        """
        orientations = self.dipole_array.orientations
        image_orientation = np.stack([np.pi - orientations[:, 0], orientations[:, 1]], axis=1)


        return image_orientation
    
    def get_image_positions(self, GND_h=0):
        dipole_positions = copy.deepcopy(self.dipole_array.r0).astype("float64")
        dipole_positions[:, -1] = 2*GND_h-dipole_positions[:, -1]
        return dipole_positions


    def add_image_sources(self, GND_h = 0):
        original_dipoles = []
        image_dipoles = []

        for ii, dipole in enumerate(self.dipole_array.dipoles):
            image_dipole = copy.deepcopy(dipole)

            x_p, y_p, z_p = dipole.unit_p
            r0_x, r0_y, r0_z = dipole.r0

            if r0_z >= GND_h:
                original_dipoles.append(dipole)
                image_dipole.r0 = [r0_x, r0_y, 2*GND_h-r0_z]
                image_dipole.set_direction_with_unit_vector([x_p, y_p, -z_p])  
                image_dipoles.append(image_dipole)
        
        all_dipoles = original_dipoles + image_dipoles


        new_dipole_array = d_array.DipoleSourceArray.init_dipole_array_from_dipole_list(self.f, all_dipoles)

        new_self = copy.deepcopy(self)
        new_self.dipole_array = new_dipole_array
        new_self.green_solutions = green_sols.GreenSolutions_MagneticSubstr(new_self)
        return new_self

    def H_evaluation(method):
        def wrapper(self: 'InfCurrentLoop', *args, **kwargs):
            G = method(self,*args, **kwargs)
            sources = self.dipole_array.dipole_moments

            H = np.einsum("i...k, i...k -> ...k", G, sources)      
            self.H = field_classes.Field3D(H, self.f, self.r)
            return self.H
        return wrapper
    
    def E_evaluation(method):
        def wrapper(self: 'InfCurrentLoop', *args, **kwargs):
            G = method(self,*args, **kwargs)
            sources = self.dipole_array.dipole_moments

            E = np.einsum("i...k, i...k -> ...k", G, sources)      
            self.E = field_classes.Field3D(E, self.f, self.r)
            return self.E
        return wrapper
    
    @H_evaluation
    def get_H_uniform_space(self) -> field_classes.Field3D:
        return self.green_solutions.Ghh_uniform_space()
    
    @H_evaluation
    def get_H_image_method(self, GND_h) -> field_classes.Field3D:
        return self.green_solutions.Ghh_image_method(GND_h)
        
    @E_evaluation
    def get_E_uniform_space(self):
        return self.green_solutions.Ghe_uniform_space()
        
    @E_evaluation
    def get_E_image_method(self, GND_h) -> field_classes.Field3D:
        return self.green_solutions.Ghe_image_method(GND_h)
        
    
    def _evaluate_electric_field_curl(self):
        assert self.H is not None, "you must first calculate the magnetic field"
        self.E = self.H.curl()/(1j*self.f*2*np.pi*self.eps0*self.eps_r)
        return self.E
    
    # aliases
    # backward compatibility
    @property
    def evaluate_magnetic_field(self):
        return self.get_H_uniform_space
    
    @property
    def evaluate_magnetic_field_over_PEC(self):
        return self.get_H_image_method
    
    @property
    def evaluate_electric_field(self):
        return self.get_E_uniform_space
    
    @property
    def evaluate_electric_field_over_PEC_images(self):
        return self.get_E_image_method
    
    


class InfCurrentLoop_with_Substrate(InfCurrentLoop):
    def __init__(
        self, 
        substrate: partial_image_expansion.Substrate,
        EM_space: UniformEMSpace,
        dipole_array=None, 
        use_magnetic_charge_moments=False,
        ):

        super().__init__(EM_space, dipole_array, use_magnetic_charge_moments)
        self.substrate = substrate

    
    @property
    def green_solutions(self):
        return green_sols.GreenSolutions_MagneticSubstr(self)
    
    def H_evaluation(method):
        def wrapper(self: 'InfCurrentLoop_with_Substrate', *args, **kwargs):
            G = method(self,*args, **kwargs)
            sources = self.dipole_array.dipole_moments

            H = np.einsum("i...k, i...k -> ...k", G, sources)      
            self.H = field_classes.Field3D(H, self.f, self.r)
            return self.H
        return wrapper
    
    def E_evaluation(method):
        def wrapper(self: 'InfCurrentLoop_with_Substrate', *args, **kwargs):
            G = method(self,*args, **kwargs)
            sources = self.dipole_array.dipole_moments

            E = np.einsum("i...k, i...k -> ...k", G, sources)      
            self.E = field_classes.Field3D(E, self.f, self.r)
            return self.E
        return wrapper
    
    @H_evaluation
    def get_H_partial_image_expansion_epsr(self, GND_h= None, N=15)-> field_classes.Field3D:
        if GND_h is None:
            GND_h = -self.substrate.thickness
        return self.green_solutions.Ghh_partial_image_expansion_epsr(GND_h, N)

    @E_evaluation   
    def get_E_partial_image_expansion(self, GND_h=None, N=15)-> field_classes.Field3D:
        if GND_h is None:
            GND_h = -self.substrate.thickness
        return self.green_solutions.Ghe_partial_image_expansion(GND_h, N)
        

    @E_evaluation
    def evaluate_electric_field_over_PEC_interpolated(self, GND_h=None, xi=0.5, N=15):
        if GND_h is None:
            GND_h = -self.substrate.thickness
        return self.green_solutions.Ghe_directly_interpolated_btw_image_methods(
            GND_h=GND_h, 
            xi=xi,
            N=N)
        
    def get_E_deconstruced_interpolation(self, GND_h=None, N=15) -> field_classes.Field3D:
        if GND_h is None:
            GND_h = -self.substrate.thickness
        G = self.green_solutions.Ghe(GND_h, N=N)
        dx, dy, dz = self.dipole_array.decompose_array()
        sources = (dx+dy+dz).dipole_moments

        E = np.einsum("n...f, nf->...f", G, sources)
        self.E = field_classes.Field3D(E, self.f, self.r)
        return self.E
    
    # aliases
    # ease of use
    
    def evaluate_E(self, *args, method="interpolated", **kargs)-> Field3D:
        """method can be 'interpolated', 'uniform space', 'image method' or 'partial image expansion'."""
        if method == "interpolated":
            return self.get_E_deconstruced_interpolation(*args, **kargs)
        if method in ["uniform space", "uniform_space", "us"]:
            return self.get_E_uniform_space(*args, **kargs)
        if method in ["image method", "image_method", "im"]:
            return self.get_E_image_method(*args, **kargs)
        if method in ["partial image expansion", "partial_image_expansion", "pie"]:
            return self.get_E_partial_image_expansion(*args, **kargs)
    
    def evaluate_H(self, *args, method="image method", **kargs) -> Field3D:
        GND_h = -self.substrate.thickness
        """method can be 'uniform space' or 'image method'."""
        if method in ["uniform space", "uniform_space", "us"]:
            return self.get_H_uniform_space(*args, **kargs)
        if method in ["image method", "image_method", "im"]:
            my_kwargs = {"GND_h": GND_h}
            my_kwargs.update(kargs)
            return self.get_H_image_method(*args, **my_kwargs)
        if method in ["partial image expansion epsr", "pie_epsr"]:
            return self.get_H_partial_image_expansion_epsr(*args, **kargs)
    
    def get_handler_for_single_dipole(self, orientation: str, *args, **kargs)->"InfCurrentLoop_with_Substrate":
        x, y = self.EM_space.r.x, self.EM_space.r.y

        x0 = (x[0]+x[-1])/2; y0 = (y[0]+y[-1])/2
        r0 = np.array([x0, y0]) 

        my_kwargs = {"r0": r0}
        my_kwargs.update(kargs)

        single_dipole_array = self.dipole_array.get_single_test_dipole_on_array(orientation, type="Magnetic", *args, **my_kwargs)
        return InfCurrentLoop_with_Substrate(self.substrate, self.EM_space, single_dipole_array, magnetic_current_source=self.magnetic_current_source)

    # backward compatibility
    @property
    def evaluate_magnetic_field_with_partial_im_exp(self):
        return self.get_H_partial_image_expansion_epsr
    @property
    def evaluate_electric_field_with_partial_im_exp(self):
        return self.get_E_partial_image_expansion

    @property
    def evaluate_electric_field_over_PEC(self):
        return self.get_E_deconstruced_interpolation
        


class DFHandler_over_Substrate(Dipole_Field_Base):
    def __init__(self, EM_space: UniformEMSpace, substrate: partial_image_expansion.Substrate, dipole_array: d_array.DipoleSourceArray):
        super().__init__(EM_space, dipole_array)
        
        if isinstance(dipole_array, d_array.FlatDipoleArray):
            my_darray_type = d_array.FlatDipoleArray
            self.gnd_height = -substrate.thickness
        elif isinstance(dipole_array, d_array.DipoleSourceArray):
            print(type(dipole_array))
            my_darray_type = d_array.DipoleSourceArray
        else:
            raise Exception("The dipole array object is of the wrong type")
        
        self.substrate = substrate

        self._electric_array = my_darray_type.init_dipole_array_from_dipole_list(
            self.dipole_array.f, [d for d in self.dipole_array.dipoles if d.type=="Electric"]).self_update_to_decomposed()
        

        self._magnetic_array = my_darray_type.init_dipole_array_from_dipole_list(
            self.dipole_array.f, [d for d in self.dipole_array.dipoles if d.type=="Magnetic"]).self_update_to_decomposed()

        if my_darray_type == d_array.FlatDipoleArray:
            self._electric_array.height = dipole_array.height
            self._magnetic_array.height = dipole_array.height
        
        self._dh_electric = InfElectricDipole_with_Substrate(EM_space=EM_space, substrate=substrate, dipole_array=self._electric_array)
        self._dh_magnetic = InfCurrentLoop_with_Substrate(EM_space=EM_space, substrate=substrate, dipole_array=self._magnetic_array)

        self._dipole_array = self.electric_array+self.magnetic_array
        if my_darray_type == d_array.FlatDipoleArray:
            self._dipole_array.height = dipole_array.height

        
    
    @property
    def dipole_array(self):
        return self._dipole_array
    
    @dipole_array.setter
    def dipole_array(self, array):
        self.__init__(EM_space=self.EM_space, substrate=self.substrate, dipole_array=array)

    @property
    def electric_array(self):
        return self._electric_array
    
    @electric_array.setter
    def electric_array(self, other: d_array.DipoleSourceArray):
        new_total_array = self.magnetic_array+other
        self.__init__(EM_space=self.EM_space, substrate=self.substrate, dipole_array=new_total_array)


    @property
    def magnetic_array(self):
        return self._magnetic_array
    
    @magnetic_array.setter
    def magnetic_array(self, other: d_array.DipoleSourceArray):
        new_total_array = self.electric_array+other
        self.__init__(EM_space=self.EM_space, substrate=self.substrate, dipole_array=new_total_array)
    
    @property
    def dh_electric(self):
        return self._dh_electric

    
    @property
    def dh_magnetic(self):
        return self._dh_magnetic
    
    def __floordiv__(self, other: d_array.DipoleSourceArray | d_array.FlatDipoleArray):

        assert isinstance(other, d_array.DipoleSourceArray) or isinstance(other, d_array.FlatDipoleArray), "The other object is not a dipole array"
        
        return DFHandler_over_Substrate(self.EM_space, self.substrate, other)


    def G_components(self, gnd_height=None, N=15):
        if gnd_height is None:
            gnd_height=self.gnd_height
        if self.dh_electric.dipole_array.N_dipoles > 0:
            Gee = self.dh_electric.green_solutions.Gee(GND_h=gnd_height, N=N)
            Geh = self.dh_electric.green_solutions.Geh(GND_h=gnd_height, N=N)
        else:
            Gee, Geh = None, None
        if self.dh_magnetic.dipole_array.N_dipoles > 0:
            Ghh = self.dh_magnetic.green_solutions.Ghh(GND_h = gnd_height)
            Ghe = self.dh_magnetic.green_solutions.Ghe(GND_h=gnd_height, N=N)
        else:
            Ghe, Ghh = None, None

        return dict(Gee=Gee, Geh=Geh, Ghh=Ghh, Ghe=Ghe)

    
    def get_G(self, gnd_height=None, N=15):
        
        """the function will consider e dipoles first then h dipoles"""
        G_comp = self.G_components(gnd_height, N)
        
        if G_comp.get("Gee") is not None: 
            Ge = np.stack([G_comp["Gee"], G_comp["Geh"]], axis=1)
        else:
            Ge = None
        if G_comp.get("Ghh") is not None:
            Gh = np.stack([G_comp["Ghe"], G_comp["Ghh"]], axis=1)
        else: 
            Gh = None
        
        valid_G = [G for G in [Ge, Gh] if G is not None]
        
        if len(valid_G) == 2:
            return np.concatenate(valid_G, axis=0)
        else:
            return valid_G[0]
    
    def evaluate_fields(self, gnd_height=None, N=15):
        if gnd_height is None:
            gnd_height=self.gnd_height
        G = self.get_G(gnd_height, N)
        sources = self.dipole_array.dipole_moments
        E, H = np.einsum("nF...f, nf->F...f", G, sources)
        self.E = field_classes.Field3D(E, self.f, self.r)
        self.E.__name__ = "E"
        self.H = field_classes.Field3D(H, self.f, self.r)
        self.H.__name__ = "H"
        return self
    
    def get_handler_for_single_dipole(self, orientation: str, dipole_type: str ="Electric", *args, **kargs)-> "DFHandler_over_Substrate":
        if orientation not in ["x", "y", "z"]:
            raise Exception("The orientation must be one of 'x', 'y', 'z'")
        if dipole_type not in ["Electric", "Magnetic"]:
            raise Exception("The dipole type must be one of 'Electric', 'Magnetic'")
        if dipole_type == "Electric":
            darray = self.dh_electric.get_handler_for_single_dipole(orientation, *args, **kargs).dipole_array
        else:
            darray = self.dh_magnetic.get_handler_for_single_dipole(orientation, *args, **kargs).dipole_array
        return DFHandler_over_Substrate(self.EM_space, self.substrate, darray)
        
    def update_dipole_array(self, new_dipole_array):
        return DFHandler_over_Substrate(self.EM_space, self.substrate, new_dipole_array)

    def update_M(self, new_M):
        new_dipole_array = self.dipole_array//new_M
        return DFHandler_over_Substrate(self.EM_space, self.substrate, new_dipole_array)
    
    def save_json(self, filepath):
        with open(filepath, "w") as f:
            json.dump(self, f, indent=4, cls=MyJsonEncoder)


    @staticmethod
    def load_json(filepath):
        with open(filepath, "r") as f:
            self = SimpleNamespace(**json.load(f))

            em_space = UniformEMSpace(
                r = self.EM_space["r"], 
                eps_r = self.EM_space["eps_r"], 
                mu_r = self.EM_space["mu_r"]
                )
            
            substrate = Substrate(
                eps_r = self.substrate["eps_r"],
                mu_r = self.substrate["mu_r"],
                thickness = self.substrate["thickness"],
                x_size = self.substrate["x_size"],
                y_size = self.substrate["y_size"],
                material_name = self.substrate["material_name"]
                )
            
            r0 = np.stack(self.dipoles["r0"])
            assert len(set(r0[:, -1]))==1, "Dipoles must be on the same z plane"
            moments = np.array(self.dipoles["M"])*np.exp(1j*np.array(self.dipoles["phases"]))

            dipole_array = d_array.FlatDipoleArray(
                r0 = r0,
                height = r0[0, -1],
                moments = moments,
                orientations = self.dipoles["orientations"],
                f = self.f
                ).set_individual_dipole_type(self.dipoles["dipole_types"])
            
            self.dfh = DFHandler_over_Substrate(em_space, substrate, dipole_array)
            print("Loaded from {}".format(filepath))
            return self.dfh

    