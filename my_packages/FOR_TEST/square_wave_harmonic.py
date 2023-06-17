import numpy as np
import math 


## create square wave harmonic functions 
def get_generator_func(func):
    def inner(*args_not_time, **kargs_not_time):
        return lambda t: func(t, *args_not_time, **kargs_not_time)
    return inner

@get_generator_func
def square_wave_FT(time, base_frequency, order=7):
    harmonics = np.arange(1, order+1, 2)
    components = np.array([1/h*np.sin(h*2*math.pi*base_frequency*time) for h in harmonics]).sum(axis=0)
    return components

def get_function_expression_HFSS(base_frequency, order=7):
    harmonics = np.arange(1, order+1, 2)
    components = ["1/"+str(h)+"*sin("+str(h)+"*2*pi*"+str(base_frequency)+"*time)" for h in harmonics]
    expression = "+".join(components)
    return expression

    
def get_PWL_from_function(func, time):
    values = func(time)
    data = [[t, value] for (t, value) in zip(time, values)]
    return data
