import numpy as np

import attr
from my_packages.classes import dipole_source, dipole_array, signals
from my_packages.classes.model_components import Substrate
from my_packages import spherical_coordinates


class PartialImageExpansion():
    def __init__(self, dipole_ar: dipole_array.DipoleSourceArray, substrate: Substrate, medium_epsr, GND_h):
        self.array = dipole_ar
        self.substrate = substrate
        self.medium_epsr = medium_epsr
        self.GND_h = GND_h
    
    def get_image_arrays(self, N):
        self.image_array = get_image_array(self.array, self.substrate, self.medium_epsr, self.GND_h, N)
        return self.image_array
    



def get_image_array(dip_array: dipole_array.DipoleSourceArray, substrate:Substrate, medium_epsr=1, GND_h=-1e-3, N=5):
    locations = get_partial_images_location(dip_array, substrate, GND_h, N)
    sig = get_partial_images_amplitude(dip_array, substrate, medium_epsr, N)
    unit_p = get_partial_images_orientation(dip_array, N)

    images = []
    
    for ii in range(len(dip_array.dipoles)):
        for r0, amp, u in zip(locations[ii], sig[ii], unit_p[ii]):
            im = dipole_source.Dipole_Source(r0, [0,0], signals.Frequency_Signal_Base(amp, dip_array.f))
            im.set_direction_with_unit_vector(u)
            images.append(im)
    
    return dipole_array.DipoleSourceArray.init_dipole_array_from_dipole_list(dip_array.f, images)


def get_partial_images_location(dipole_array: dipole_array.DipoleSourceArray, substrate: Substrate, GND_h=-1e-3, N=5):
    return [get_partial_images_location_4_dipole(dipole, substrate, GND_h, N) for dipole in dipole_array.dipoles]

def get_partial_images_amplitude(dipole_array: dipole_array.DipoleSourceArray, substrate: Substrate, medium_epsr=1, N=5):
    return [get_partial_image_amplitude_4_dipole(dipole, substrate, medium_epsr, N) for dipole in dipole_array.dipoles]

def get_partial_images_orientation(dipole_array: dipole_array.DipoleSourceArray, N=5):
    return [get_partial_image_orientation_4_dipole(dipole, N) for dipole in dipole_array.dipoles]
    

def get_partial_images_location_4_dipole(dipole: dipole_source.Dipole_Source, substrate: Substrate, GND_h = -1e-3, N=5):
    
    h = dipole.r0[-1] - GND_h - substrate.thickness # separation between dipole and substrate surface
    assert h>=0
    
    first_order_image_position = np.array([dipole.r0[0], dipole.r0[1], GND_h+substrate.thickness - h])

    # each higher order image is spaced exactly 2d below the previous image, with d the substrate thickness

    higher_order_images = [np.array(first_order_image_position)-[0,0,2*substrate.thickness*(ii+1)] for ii in range(N-1)]
    
    return np.stack([first_order_image_position]+higher_order_images, axis=0)

def get_partial_image_expansion_coefficients_4_dipole(substrate:Substrate, medium_epsr=1, N=5):    
    gamma = refl_coefficient(medium_epsr, substrate.eps_r) #
    first_order_image = [gamma] 

    higher_order_image = [-((1-gamma)*(1+gamma)*(gamma)**ii) for ii in range(0, N-1)]


    return np.stack(first_order_image+higher_order_image, axis=0)


def get_partial_image_orientation_4_dipole(dipole: dipole_source.Dipole_Source, N=5):
    # Consider the dipole as the due component charges. 
    # it should be that at 90deg the image inverts orientation while flat dipoles maintain the same orientation.
    orientations = [dipole.unit_p*np.array([1,1,-1])]*N
    return orientations


def get_partial_image_amplitude_4_dipole(dipole: dipole_source.Dipole_Source, substrate:Substrate, medium_epsr=1, N=5):
    moment_signal = dipole.dipole_moment.signal

    gamma = refl_coefficient(medium_epsr, substrate.eps_r) #
    first_order_image = [gamma*moment_signal] 

    higher_order_image = [((1-gamma)*(1+gamma)*(gamma)**ii)*moment_signal for ii in range(0, N-1)]


    return np.stack(first_order_image+higher_order_image, axis=0)

def refl_coefficient(eps_r0, eps_r):
    return (eps_r0-eps_r)/(eps_r0+eps_r)


