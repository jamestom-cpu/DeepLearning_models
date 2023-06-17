import numpy as np
import pandas as pd
from typing import Iterable
from my_packages.classes import signals
from my_packages import spherical_coordinates
import copy

from typing import Iterable

class Dipole_Source():
    def __init__(self, position, orientation=[0,0], source=None, domain = "Frequency", type=None):
        self.r0 = position
        self.theta = orientation[0]
        self.phi = orientation[1]
        # unit vector that point in the direction of the dipole
        self._domain = domain
        self.dipole_moment = source
        self.type = type
    
    def set_direction_with_unit_vector(self, p):
        _, self.theta, self.phi = spherical_coordinates.to_spherical_grid(p)
        
    @property
    def unit_p(self):
        return np.array([np.sin(self.theta)*np.cos(self.phi), np.sin(self.theta)*np.sin(self.phi), np.cos(self.theta)], dtype=np.float64)




    @property
    def domain(self):
        return self._domain
    
    @domain.setter
    def domain(self):
        print("run the change_domain() function to change domain")


   
    
    

    @property
    def dipole_moment(self):
        return self._dipole_moment

    @dipole_moment.setter
    def dipole_moment(self, dipole_signal):
        
        if self.domain == "Frequency":
            assert type(dipole_signal) in [signals.Frequency_Signal, signals.Frequency_Signal_Base, type(None)]
            self._dipole_moment = dipole_signal
        if self.domain == "Time":
            assert type(dipole_signal) in [signals.Time_Signal, type(None)]
            self._dipole_moment = dipole_signal
    
    def decompose_dipole(self):
        moment = self.dipole_moment.signal # shape f
        moment_z = moment*np.cos(self.theta)
        moment_y = moment*np.sin(self.theta)*np.sin(self.phi)
        moment_x = moment*np.sin(self.theta)*np.cos(self.phi)

        lst = [moment_x, moment_y, moment_z]
        angles = [[np.pi/2, 0], [np.pi/2, np.pi/2], [0, 0]]
        
        def not_close_to_zero(member, threshold):
            return (abs(member) > abs(threshold)).any() and (member!= 0).any()
        
        output_dipoles = []
        for m, ors in zip(lst, angles): 
            if not_close_to_zero(m, np.abs(np.array(lst)).max()/100):
                signal = signals.Frequency_Signal_Base(signal=m, f=self.dipole_moment.f)
                dec_dip = Dipole_Source(self.r0, ors, signal)
                output_dipoles.append(dec_dip)
            
            else:
                output_dipoles.append(0)
        return output_dipoles
    
    def transform_moment_to_signal(self, mult_factor=1, f_Nyquist=None, f_min = np.finfo(np.float64).eps):
        assert(self.domain == "Frequency")
        self.dipole_moment = self.dipole_moment.transform_to_signal(mult_factor=mult_factor, f_Nyquist=f_Nyquist, f_min=f_min)
        return self

    def get_overall_source_matrix(self):
        assert self.domain == "Frequency"
        self.source_matrix = np.outer(self.unit_p, self.dipole_moment.signal)
        return self.source_matrix

    def change_domain(self):
        new_self = copy.deepcopy(self)
        my_dict = {"Time":True, "Frequency":False}
        inverse_dict = {1: "Time", 0: "Frequency"}

        current_domain = my_dict[self.domain]
        new_domain = inverse_dict[int(not current_domain)]

        # current domain is true if in the time domain
        time_domain = current_domain
        if time_domain:
            # we are in the frequency domain
            freq_domain_obj = new_self.dipole_moment.to_frequency_domain()
            new_self._dipole_moment = freq_domain_obj
            new_self._domain = "Frequency"
            new_self._f_Nyquist = new_self.dipole_moment.f_Nyquist

        else:
            assert type(self.dipole_moment) is signals.Frequency_Signal
            time_domain_obj = new_self.dipole_moment.to_time_domain()
            new_self._dipole_moment = time_domain_obj
            new_self._domain = "Time"
            new_self._interval_duration = new_self.dipole_moment.T
        
        return new_self
    



    
    

    
    
