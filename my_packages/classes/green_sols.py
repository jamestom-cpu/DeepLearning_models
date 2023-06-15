
import numpy as np
from my_packages import field_calculators
from my_packages.protocols import DipoleArr_Prot, DipoleFC_Prot, ElectricSubstrFC_Prot, MagneticFC_Prot
from my_packages import partial_image_expansion, spherical_coordinates

class GreenSolutions():
    def __init__(self, dipole_fc: DipoleFC_Prot):
        self.dipole_fc = dipole_fc
        self.k = dipole_fc.k
        self.r = dipole_fc.r
        self.r0 = dipole_fc.dipole_array.r0
        self.orientations=dipole_fc.dipole_array.orientations
        # self.eps_r = dipole_fc.eps_r
        # self.mu_r = dipole_fc.mu_r
        # self.sigma = dipole_fc.sigma
    
    def dyadic_for_vector_moments_single_dipole(self, r0, scale_with_wave_number=True):
        return field_calculators.magnetic_greens_functions_solution_dyad(k=self.k, r=self.r, r0=r0, scale_with_wave_number=scale_with_wave_number)
    
    def for_set_single_dipole_orientation(self, r0, orientations, scale_with_wave_number = True):
        return field_calculators.magnetic_greens_solution_for_moment_amplitude(k=self.k, r=self.r, r0=r0, orientations=orientations, scale_with_wave_number=scale_with_wave_number)
    
    def Gdyadic_uniform_space(self, r0=None, scale_with_wave_number=True):
        if r0 is None:
            r0 = self.r0
        green_sols = [self.dyadic_for_vector_moments_single_dipole(position, scale_with_wave_number) for position in r0]
        return np.stack(green_sols, axis=0)
    
    def G_uniform_space_base(self, r0=None, orientations=None, scale_with_wave_number=True):
        if r0 is None:
            r0 = self.r0
        if orientations is None:
            orientations = self.orientations
        green_sols = [self.for_set_single_dipole_orientation(position, orients, scale_with_wave_number) for position, orients in zip(r0, orientations)]
        return np.stack(green_sols, axis=0)


class GreenSolutions_MagneticSubstr(GreenSolutions):
    def __init__(self, dipole_fc: MagneticFC_Prot):
        super().__init__(dipole_fc)
        self.f = dipole_fc.f
        self.mu0 = dipole_fc.mu0
        self.mu_r = dipole_fc.mu_r
        self.eps_r = dipole_fc.eps_r
        self.dipole_array = dipole_fc.dipole_array
        self.r = dipole_fc.r
        
        if hasattr(dipole_fc, "substrate"):
            self.substrate = dipole_fc.substrate


        self.keep_valid_dipoles = dipole_fc.keep_valid_dipoles
        self.get_image_positions = dipole_fc.get_image_positions
        self.get_image_orientations = dipole_fc.get_image_orientations

    def Ghh_uniform_space(self, r0=None, orientations=None) -> np.array:
        return self.G_uniform_space_base(r0=r0, orientations=orientations, scale_with_wave_number=True)

    def Ghe_uniform_space(self, r0=None, orientations=None)-> np.ndarray:

        if orientations is None:
            unit_p = [dipole.unit_p for dipole in self.dipole_array.dipoles]
        else:
            unit_p = spherical_coordinates.unit_direction_from_theta_phi(orientations)

        my_kargs = dict(
            f = self.f,
            k = self.k,
            mu = self.mu0*self.mu_r,
            unit_m = unit_p,
            r = self.r,
            r0 = [dipole.r0 for dipole in self.dipole_array.dipoles] if r0 is None else r0
        )
        return field_calculators.inf_loop_G_electric(**my_kargs)
    
    def Ghe_directly_interpolated_btw_image_methods(self, GND_h: float, xi=0.5, N=15)-> np.ndarray:
        """
        xi is a float term that varies between 0 and 1.
        xi = 0 returns Gpie
        xi = 1 returns Gim
        """
        Gim = self.Ghe_image_method(GND_h)
        Gpie = self.Ghe_partial_image_expansion(GND_h, N)
        return Gpie + (Gim-Gpie)*xi

    def Ghe_image_method(self, GND_h):
        orientations = self.get_image_orientations()
        positions = self.get_image_positions(GND_h)
        image_green_solution = self.Ghe_uniform_space(r0=positions, orientations=orientations)
        free_space_solution = self.Ghe_uniform_space()

        return image_green_solution+free_space_solution
    
    def Ghh_partial_image_expansion_epsr(self, GND_h, N=5):
        coeffs = partial_image_expansion.get_partial_image_expansion_coefficients_4_dipole(self.substrate, medium_epsr=self.eps_r, N=N)
        locs = np.array(partial_image_expansion.get_partial_images_location(self.dipole_array, self.substrate, GND_h=GND_h, N=N))
        unit_p = -np.array(partial_image_expansion.get_partial_images_orientation(self.dipole_array, N=N))

        G_list = [self.Ghh_uniform_space()]

        for ii in range(N):
            r0 = locs[:,ii,:]
            u = unit_p[:,ii,:]
            _, theta, phi = spherical_coordinates.to_spherical_grid(u.T)
            ors = np.stack([theta, phi], axis=1)
            Gii = self.Ghh_uniform_space(r0=r0, orientations=ors)*coeffs[ii]
            G_list.append(Gii)

        Gsum = np.sum(G_list, axis=0)
        return Gsum

    def Ghe_partial_image_expansion(self, GND_h, N=5):
        coeffs = partial_image_expansion.get_partial_image_expansion_coefficients_4_dipole(self.substrate, N=N)
        locs = np.array(partial_image_expansion.get_partial_images_location(self.dipole_array, self.substrate, GND_h=GND_h, N=N))
        unit_p = -np.array(partial_image_expansion.get_partial_images_orientation(self.dipole_array, N=N)) 

        
        G_list = [self.Ghe_uniform_space()]

        for ii in range(N):
            r0 = locs[:,ii,:]
            u = unit_p[:,ii,:]
            _, theta, phi = spherical_coordinates.to_spherical_grid(u.T)
            ors = np.stack([theta, phi], axis=1)
            Gii = self.Ghe_uniform_space(r0=r0, orientations=ors)*coeffs[ii]
            G_list.append(Gii)

        Gsum = np.sum(G_list, axis=0)
        return Gsum


    def Ghh_image_method(self, GND_h):
        orientations = self.get_image_orientations()
        positions = self.get_image_positions(GND_h)
        image_green_solution = self.Ghh_uniform_space(r0=positions, orientations=orientations)
        free_space_solution = self.Ghh_uniform_space()

        return image_green_solution+free_space_solution



    def Ghh_dyadic_images(self, GND_h):
        self.keep_valid_dipoles(GND_h)
        positions = self.get_image_positions(GND_h)

        image_green_solution = self.Gdyadic_uniform_space(r0=positions)
        # the orientations are not included, so we must add a sign change
        image_green_solution = np.einsum("i, ki...-> ki...", np.array([1,1,-1]), image_green_solution)
        free_space_solution = self.Gdyadic_uniform_space()
        return image_green_solution + free_space_solution
    
    def Ghe_decomposed_and_interpolated(self, GND_h: float, N=15):
        dx, dy, dz = self.dipole_array.decompose_array()
        dflat = dx+dy

        gsol_norm = GreenSolutions_MagneticSubstr(self.dipole_fc//dz)
        gsol_flat = GreenSolutions_MagneticSubstr(self.dipole_fc//dflat)
        
        Gflat = gsol_flat.Ghe_directly_interpolated_btw_image_methods(GND_h=GND_h, xi=0.5, N=N) if dflat.N_dipoles > 0 else None
        Gnorm = gsol_norm.Ghe_directly_interpolated_btw_image_methods(GND_h=GND_h, xi=0.95, N=N) if dz.N_dipoles > 0 else None

        return np.concatenate([G for G in [Gflat, Gnorm] if G is not None], axis=0)
    
    # aliases for simplicity
    @property
    def Ghh(self)-> callable:
        return self.Ghh_image_method
    @property
    def Ghe(self) -> callable:
        return self.Ghe_decomposed_and_interpolated

    
class GreenSolutions_ElectricSubstr(GreenSolutions):
    def __init__(self, dipole_fc: ElectricSubstrFC_Prot):
        super().__init__(dipole_fc)
        self.f = dipole_fc.f
        self.eps0 = dipole_fc.eps0
        self.eps_r = dipole_fc.eps_r
        self.dipole_array = dipole_fc.dipole_array
        if hasattr(dipole_fc, "substrate"):
            self.substrate = dipole_fc.substrate
        self.keep_valid_dipoles = dipole_fc.keep_valid_dipoles
        self.get_image_orientations = dipole_fc.get_image_orientations
        self.get_image_positions = dipole_fc.get_image_positions
    

    
    def Geh_uniform_space(self, r0=None, orientations=None):
        if orientations is None:
            unit_p = [dipole.unit_p for dipole in self.dipole_array.dipoles]
        else:
            unit_p = spherical_coordinates.unit_direction_from_theta_phi(orientations)
        
        my_kargs = dict(
            k = self.k,
            unit_p =  unit_p,
            r = self.r,
            r0 = [dipole.r0 for dipole in self.dipole_array.dipoles] if r0 is None else r0
        )
        return field_calculators.electric_dipole_G_magnetic(**my_kargs)
    
    def Geh_image_method(self, GND_h):
        orientations = self.get_image_orientations()
        positions = self.get_image_positions(GND_h)
        image_green_solution = self.Geh_uniform_space(r0=positions, orientations=orientations)
        free_space_solution = self.Geh_uniform_space()

        return image_green_solution+free_space_solution


    def Gee_uniform_space(self, r0=None, orientations=None, scale_with_wave_number=True):
        return super().G_uniform_space_base(r0, orientations, scale_with_wave_number)/(1j*2*np.pi*self.f*self.eps0*self.eps_r)


    def Gee_image_method(self, GND_h):
        orientations = self.get_image_orientations()
        positions = self.get_image_positions(GND_h)
        image_green_solution = self.Gee_uniform_space(r0=positions, orientations=orientations)
        free_space_solution = self.Gee_uniform_space()

        return image_green_solution+free_space_solution

    def Gee_partial_image_expansion(self, GND_h, N=5):
        coeffs = partial_image_expansion.get_partial_image_expansion_coefficients_4_dipole(self.substrate, medium_epsr=self.eps_r, N=N)
        locs = np.array(partial_image_expansion.get_partial_images_location(self.dipole_array, self.substrate, GND_h=GND_h, N=N))
        unit_p = np.array(partial_image_expansion.get_partial_images_orientation(self.dipole_array, N=N)) 

        
        G_list = [self.Gee_uniform_space()]

        for ii in range(N):
            r0 = locs[:,ii,:]
            u = unit_p[:,ii,:]
            _, theta, phi = spherical_coordinates.to_spherical_grid(u.T)
            ors = np.stack([theta, phi], axis=1)
            Gii = self.Gee_uniform_space(r0=r0, orientations=ors)*coeffs[ii]
            G_list.append(Gii)

        Gsum = np.sum(G_list, axis=0)
        return Gsum
    
    def Geh_partial_image_expansion(self, GND_h, N=5):
        coeffs = partial_image_expansion.get_partial_image_expansion_coefficients_4_dipole(self.substrate, N=N)
        locs = np.array(partial_image_expansion.get_partial_images_location(self.dipole_array, self.substrate, GND_h=GND_h, N=N))
        unit_p = np.array(partial_image_expansion.get_partial_images_orientation(self.dipole_array, N=N)) 

        
        G_list = [self.Geh_uniform_space()]

        for ii in range(N):
            r0 = locs[:,ii,:]
            u = unit_p[:,ii,:]
            _, theta, phi = spherical_coordinates.to_spherical_grid(u.T)
            ors = np.stack([theta, phi], axis=1)
            Gii = self.Geh_uniform_space(r0=r0, orientations=ors)*coeffs[ii]
            G_list.append(Gii)

        Gsum = np.sum(G_list, axis=0)
        return Gsum
    
    def Geh_directly_interpolated_btw_image_methods(self, GND_h, xi=0.5, N=15):
        """
        xi is a float term that varies between 0 and 1.
        xi = 0 returns Gpie
        xi = 1 returns Gim
        """
        Gim = self.Geh_image_method(GND_h) # quasistatic approx
        Gpie = self.Geh_partial_image_expansion(GND_h, N) # Maxwell relationship

        # the roles of Gpie and Gim are inverted as the 2 approximations they represent are inverted
        return Gim + (Gpie-Gim)*xi
    
    def Geh_decomposed_and_interpolated(self, GND_h: float, N=15):
        dx, dy, dz = self.dipole_array.decompose_array()
        dflat = dx+dy

        gsol_norm = GreenSolutions_ElectricSubstr(self.dipole_fc//dz)
        gsol_flat = GreenSolutions_ElectricSubstr(self.dipole_fc//dflat)
        
        Gflat = gsol_flat.Geh_directly_interpolated_btw_image_methods(GND_h=GND_h, xi=0.5, N=N) if dflat.N_dipoles > 0 else None
        Gnorm = gsol_norm.Geh_directly_interpolated_btw_image_methods(GND_h=GND_h, xi=0.95, N=N) if dz.N_dipoles > 0 else None

        return np.concatenate([G for G in [Gflat, Gnorm] if G is not None], axis=0)

    @property
    def Gee(self)-> callable:
        return self.Gee_partial_image_expansion
    @property
    def Geh(self)-> callable:
        return self.Geh_decomposed_and_interpolated