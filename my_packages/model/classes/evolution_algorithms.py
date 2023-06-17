import numpy as np
from dataclasses import dataclass, field
from collections import namedtuple
from typing import Tuple, List, Self, Iterable
from enum import Enum
import random
from functools import partial
import threading
from queue import PriorityQueue

from my_packages.classes.model_components import UniformEMSpace, Substrate
from my_packages.classes.field_classes import Field3D
from my_packages.classes.dipole_array import FlatDipoleArray
from my_packages.classes.dipole_fields import DFHandler_over_Substrate


from .evolution_helper_classes import DipoleLayoutPopulation, Mask
from .optimer_amp import OptimalSourcesValues

RegCoeffs = namedtuple("RegCoeffs", ["E", "H"])
MutationChances = namedtuple("MutationChances", ["N", "p"])


@dataclass
class Elements():
    genomes: List[Mask]
    fitness_score: List[float]
    
    def __call__(self):
        return list(zip(self.fitness_score, self.genomes))

    def sort(self):
        element_list = self()
        sorted_list = sorted(element_list, key=lambda x: x[0], reverse=True)

        sorted_fitness, sorted_genomes = np.array(sorted_list, dtype="object").T
        return Elements(sorted_genomes, sorted_fitness)
    def mate_solutions(self)-> Tuple[List[float], List[float]]:
        return tuple(random.choices(population=self.genomes, weights=self.fitness_score, k=2))
        




@dataclass
class GeneticAlg_Operator:
    EM_space: UniformEMSpace
    substr: Substrate
    E_target: Field3D
    H_target: Field3D
    population: DipoleLayoutPopulation
    _opt_type: str = "Magnetic"
    min_loss: float = 0
    max_fitness: float = 0
    # elements is initialized when the fitness is evaluated
    elements: Elements = field(default_factory=Elements([], [])) 
    
    @property
    def opt_type(self):
        return self._opt_type
    
    @opt_type.setter
    def opt_type(self, value: str):
        if value not in ["Magnetic", "Electric"]:
            raise ValueError("opt_type should be either 'Magnetic' or 'Electric'")
        self.population.update_dipole_type(value)
        self._opt_type = value 


    def update_population(self, new_masks: List[Mask])-> Self:
        self.population = self.population.update_population(new_masks)
        return self

    def get_new_generation(
        self, elitism_proportion: float = 0.2, N_crossovers: int=2, 
        mutation_parameters: Tuple[int, float]=(30, 0.2)) -> Self:

        if isinstance(elitism_proportion, float):
            offspring_proportion = 1-elitism_proportion
        elif isinstance(elitism_proportion, int):
            offspring_proportion = len(self.population)-elitism_proportion
        else:
            raise AttributeError("elitism_proportion should be either an int or a float")
        
        elite_solutions = self.select_best_samples(elitism_proportion)
        couple_soultions = self.couple_solutions(
            offspring=offspring_proportion, 
            N_crossovers=N_crossovers, mutation_params=mutation_parameters)

        new_genomes = elite_solutions+couple_soultions
        return self.update_population(new_genomes)

    def couple_solutions(
        self, offspring: float = 0.7, N_crossovers: int = 2, 
        mutation_params: Tuple[int, float]=(30, 0.2)):
        if type(offspring) is float:
            index = int((len(self.population)*offspring))
        elif type(offspring) is int:
            index = offspring
        else:
            raise TypeError("elitism should be a float or an int")
        
        new_generation = []
        
        for ii in range(int((index+1)/2)):
            parents = self.elements.mate_solutions()
            if N_crossovers == 1:
                offspring_a, offspring_b = GeneticAlg_Operator.single_point_crossover(*parents)
            if N_crossovers>1:
                offspring_a, offspring_b = GeneticAlg_Operator.crossover(*parents, N_crossovers-1)

            offspring_a = GeneticAlg_Operator.mutation(offspring_a, *mutation_params)
            offspring_b = GeneticAlg_Operator.mutation(offspring_b, *mutation_params)

            new_generation += [offspring_a, offspring_b]

        return new_generation

    def select_best_samples(self, elitism: float=0.3):
        if type(elitism) is float:
            index = int(np.floor(len(self.population)*elitism))
        elif type(elitism) is int:
            index = elitism
        else:
            raise TypeError("elitism should be a float or an int")
        
        return list(self.elements.genomes[:index])
    
    def generate_best_dipole_source_array(self):
        top_mask = self.elements.genomes[0]
        cell_grid = self.population.cell_grid.update_mask(top_mask)
        cell_array = cell_grid.ActiveArrays.all

        f = self.population.dipole_array_ref.freqs
        height = self.population.dipole_array_ref.height
        darray = cell_array.generate_dipole_array(height=height, f=f).set_dipole_type(self.opt_type)
        return darray

    def generate_best_fh(self, reg_coefficient:float = 1e-4):
        darray = self.generate_best_dipole_source_array()
        d_fh = DFHandler_over_Substrate(EM_space=self.EM_space, substrate=self.substr, dipole_array=darray)
        opt_source = OptimalSourcesValues(
            fh=d_fh, measured_E=self.E_target, 
            measured_H=self.H_target, reg_coefficient=reg_coefficient)
        
        return opt_source.optimize_magnetic_dipoles().fh
            

    @staticmethod
    def mutation(genome: Mask, N_chances: int = 1, probability: float = 0.5) -> np.ndarray:

        a_flat = genome.flatten()
        original_shape = genome.shape

        for _ in range(N_chances):
            index = random.randrange(len(a_flat))
            a_flat[index] = a_flat[index] if random.random() > probability else not a_flat[index] 
            
        return a_flat.reshape(original_shape)


    @staticmethod
    def crossover(a: Mask, b: Mask, N_crossovers: int) -> Tuple[Mask, Mask]:
        assert a.shape == b.shape, AssertionError("the shapes of the genomes must be the same")
        original_shape = a.shape

        a_flat = a.flatten()
        b_flat = b.flatten()

        crossover_points = sorted(random.sample(range(1, len(a_flat)+1), N_crossovers))
        broken_a= break_up_array(a_flat, crossover_points=crossover_points)
        broken_b= break_up_array(b_flat, crossover_points=crossover_points)

        # list of [N, 2] elements
        # each piece has different length so we cannot use numpy stack
        crossed_list = [
            GeneticAlg_Operator.single_point_crossover(section_a, section_b) 
            for section_a, section_b in zip(broken_a, broken_b)
            ]

        a_chunks, b_chunks = np.array(crossed_list, dtype="object").T
        new_a = np.concatenate(a_chunks).reshape(original_shape).view(Mask)
        new_b = np.concatenate(b_chunks).reshape(original_shape).view(Mask)
        return new_a, new_b
        
        
        
        
    
    
    @staticmethod  
    def single_point_crossover(a:Mask, b:Mask):
        a_flat = a.flatten(); b_flat=b.flatten()

        assert a.shape==b.shape, AssertionError("a and b must have the same length")
        length = len(a_flat)
        shape = a.shape

        if length<2:
            return a,b
        
        p = random.randint(1, length-1)
        offspring1 = np.concatenate([a_flat[0:p], b_flat[p:]]).reshape(shape).view(Mask)
        offspring2 = np.concatenate([b_flat[0:p], a_flat[p:]]).reshape(shape).view(Mask)
        return offspring1, offspring2

    def _single_thread_func(self, darray_list: List[FlatDipoleArray], thread_number:int, result_queue: PriorityQueue, reg_loss_func: float, reg_coefficient_EH:float):
        # print(f"starting thread {thread_number}..")
        pop_scores = [self._evaluate_fitness(
            darr_m=darray, 
            reg_loss_func = reg_loss_func,
            reg_coefficient_EH=reg_coefficient_EH, opt_type=self.opt_type) 
            for darray in darray_list
            ]
        # print(f"thread {thread_number}, terminated ..")
        result_queue.put((thread_number, pop_scores))
    
    def _return_population_scores(self, reg_coefficient_EH: float, reg_loss_func:float=0.1, n_threads=1):
        
        if n_threads==1:
            return [self._evaluate_fitness(
                darr_m=self.population.dipole_arrays[ii], 
                reg_loss_func = reg_loss_func,
                reg_coefficient_EH=reg_coefficient_EH, opt_type=self.opt_type) 
                for ii in range(len(self.population))]

        
        dipole_arrays_per_thread = split_into_populations(self.population.dipole_arrays, n_threads)

        population_scores_queue = PriorityQueue()      
        single_thread_fitness_eval = partial(
            self._single_thread_func, 
            result_queue=population_scores_queue, 
            reg_loss_func=reg_loss_func, 
            reg_coefficient_EH=reg_coefficient_EH
            )
        threads = []

        for ii in range(n_threads):
            thread = threading.Thread(
                target=single_thread_fitness_eval, 
                kwargs={
                    "darray_list": dipole_arrays_per_thread[ii],
                    "thread_number": ii
                    }
                )
            thread.start()
            threads.append(thread)
        
        # wait for all threads to finish
        for thread in threads:
            thread.join()
        
        pop_scores = []
        while not population_scores_queue.empty():
            pop_scores += population_scores_queue.get()[1]        

        # print("lists are the same: ", np.allclose(np.array(pop_scores), np.array(pop_scores1)))

        # for (f, _), (f1, _) in zip(pop_scores, pop_scores1):
        #     print(f"{f}-{f1}")

        # print(np.array(sorted(pop_scores1, key=lambda x: x[0], reverse=True))==np.array(sorted(pop_scores, key=lambda x: x[0], reverse=True)))
        return pop_scores

    def evaluate_population_fitness(self, reg_coefficient_EH: float, reg_loss_func:float=0.1, n_threads=1):
        pop_scores = self._return_population_scores(reg_coefficient_EH=reg_coefficient_EH, reg_loss_func=reg_loss_func, n_threads=n_threads)

        # pop_scores = [self._evaluate_fitness(
        #     darr_m=self.population.dipole_arrays[ii], 
        #     reg_loss_func = reg_loss_func,
        #     reg_coefficient_EH=reg_coefficient_EH, opt_type=self.opt_type) 
        #     for ii in range(len(self.population))]

        fitness = [scores[0] for scores in pop_scores]
        mse = [scores[1] for scores in pop_scores]

        self.min_loss = min(mse)
        self.max_fitness = max(fitness)

        self.elements = Elements(self.population.masks, fitness).sort() 
        return self


    def _evaluate_fitness(self, darr_m: FlatDipoleArray, reg_loss_func: float, reg_coefficient_EH:float, opt_type="Magnetic")-> Tuple[float, float]:

        d_fh = DFHandler_over_Substrate(
            EM_space=self.EM_space, 
            substrate=self.substr, 
            dipole_array=darr_m,
            )
        
        opt_source = OptimalSourcesValues(
            fh=d_fh, measured_E=self.E_target, 
            measured_H=self.H_target, reg_coefficient=reg_coefficient_EH)

        N_active_dipoles = len(d_fh.dipole_array.dipoles)

        if opt_type=="Magnetic":
            opt_source.optimize_magnetic_dipoles().evaluate_H()
            mse = opt_source.evaluate_mse_H()
            return self.fitness_func(mse, N_active_dipoles, reg_loss_func), mse
            
        if opt_type == "Electric":
            opt_source.N_expansion=1
            opt_source.optimize_electric_dipoles().evaluate_E()
            mse = opt_source.evaluate_mse_E()
            return self.fitness_func(mse, N_active_dipoles, reg_loss_func), mse
        
    
    def fitness_func(self, mse:float, N_active_dipoles: int, loss_coefficient:float=0.1)-> float:
        return GeneticAlg_Operator.loss_func(mse, N_active_dipoles, loss_coefficient)**-1
    
    @staticmethod
    def loss_func(mse:float, N:int, coefficient:float)-> float:
        return mse + coefficient*N**2
        
def break_up_array(arr: Iterable, crossover_points: List[int])-> List[Iterable]:
    sub_arrays = []
    start = 0
    for ii in range(len(crossover_points)):
        end = crossover_points[ii]
        sub_arrays.append(arr[start:end])
        start = end
    sub_arrays.append(arr[end:])
    return sub_arrays

def split_into_populations(d_arrays: List[FlatDipoleArray], N: int)-> List[FlatDipoleArray]:
    length = len(d_arrays)
    step = length // N
    leftover = length%N

    num_el = [step + 1 if leftover>=ii else step for ii in range(1, N+1)]
    indices = [0]
    for N in num_el:
        indices.append(indices[-1]+N)
    
    split_parts = []
    for ii in range(1, len(indices)):
        split_parts.append(d_arrays[indices[ii-1]:indices[ii]])

    # leftover_vector = np.zeros(N, dtype=int)
    # leftover_vector[:leftover] = 1
    # split_parts = []
    # previous_extra = 0
    # for ii, extra in zip(range(0, length, step), leftover_vector):
    #     print("extra is",extra)
    #     split_parts.append(d_arrays[ii+previous_extra:ii+step+int(extra)])
    #     previous_extra=extra
    return split_parts