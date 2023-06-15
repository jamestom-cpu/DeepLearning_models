import numpy as np
from typing import Iterable, Tuple

class Grid(np.ndarray):

    def __new__(cls, grid: Iterable) -> None:
        if len(grid) < 2 or len(grid) > 4:
            raise ValueError("Grid must be 2 or 3 dimensional")
        obj = np.ascontiguousarray(grid).view(cls)
        return obj
    
    @property
    def v(self):
        return self.view(np.ndarray)
    
    @property
    def x(self) -> np.ndarray:
        if len(self.shape) == 4:
            return np.ascontiguousarray(self[0,:,0,0])
        elif len(self.shape) == 3:
            ax = self.check_2d_axis()
            if ax == "x":
                return np.ascontiguousarray(self[0,0,0])
            if ax == "y":
                return np.ascontiguousarray(self[0,:,0])
            if ax == "z":
                return np.ascontiguousarray(self[0,:,0])
        else:
            raise ValueError("Grid must be 2 or 3 dimensional")
    @property
    def y(self) -> np.ndarray:
        if len(self.shape) == 4:
            return np.ascontiguousarray(self[1,0,:,0])
        elif len(self.shape) == 3:
            ax = self.check_2d_axis()
            if ax == "x":
                return np.ascontiguousarray(self[1,:,0])
            if ax == "y":
                return np.ascontiguousarray(self[1,0,0])
            if ax == "z":
                return np.ascontiguousarray(self[1,0,:])
        else:
            raise ValueError("Grid must be 2 or 3 dimensional")
    @property
    def z(self) -> np.ndarray:
        if len(self.shape) == 4:
            return np.ascontiguousarray(self[2,0,0,:])
        if len(self.shape) == 3:
            ax = self.check_2d_axis()
            if ax == "x":
                return np.ascontiguousarray(self[2,0,:])
            if ax == "y":
                return np.ascontiguousarray(self[2,0,:])
            if ax == "z":
                return np.ascontiguousarray(self[2,0,0])
        else:
            raise ValueError("Grid must be 2 or 3 dimensional")
    
    def check_2d_axis(self)->str:
        if len(self.shape) != 3:
            raise ValueError("This function should be called on a 2D grid")
        axis = [ii for ii in range(3) if self[ii].min()==self[ii].max()]
        if len(axis) != 1:
            raise ValueError(f"Grid must have 2 dimensions with multiple values. Found axis: {axis} that are constant")
        axis = "xyz"[axis[0]]
        return axis

        

    def bounds(self) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
        xax, yax, zax = self.axes()
        return (float(xax.min()), float(xax.max())), (float(yax.min()), float(yax.max())), (float(zax.min()), float(zax.max()))
    
    def axes(self) -> Tuple[np.ndarray,np.ndarray,np.ndarray]:
        if len(self.shape) in [3, 4]:
            return self.x, self.y, self.z
        else:
            raise ValueError("Grid must be 2 or 3 dimensional")
        
    
    def resample_on_shape(self, shape: Tuple[int, int, int])-> "Grid":
        x, y, z = self.axes()
        xsize, ysize, zsize = shape
        return Grid._from_bounds_explicit(x.min(), x.max(), y.min(), y.max(), z.min(), z.max(), xsize, ysize, zsize)
    
    def resample_at_height(self, height: float) -> "Grid":
        x, y, z = self.axes()
        _, xsize, ysize, _ = self.shape
        return Grid._from_bounds_explicit(x.min(), x.max(), y.min(), y.max(), height, height, xsize, ysize, 1)
    
    def __array_finalize__(self, obj):
        if obj is None: return
    
    @staticmethod
    def from_bounds(xbounds: Tuple | float, ybounds: Tuple | float, zbounds: Tuple | float, xsize: int = 0, ysize: int = 0, zsize: int = 0) -> 'Grid':
        if isinstance(xbounds, float):
            xbounds = (xbounds, xbounds)
        if isinstance(ybounds, float):
            ybounds = (ybounds, ybounds)
        if isinstance(zbounds, float):
            zbounds = (zbounds, zbounds)
        
        xmin, xmax = xbounds
        ymin, ymax = ybounds
        zmin, zmax = zbounds
        return Grid._from_bounds_explicit(xmin, xmax, ymin, ymax, zmin, zmax, xsize, ysize, zsize)
    
    @staticmethod
    def _from_bounds_explicit(xmin: float, xmax: float, ymin: float, ymax: float, zmin: float, zmax: float, xsize: int = 0, ysize: int = 0, zsize: int = 0) -> 'Grid':
        if xmin>xmax:
            xmax, xmin = xmin, xmax
        
        if xmin==xmax:
            x = [xmin]
        else:
            x = np.linspace(xmin, xmax, xsize)
        
        if ymin>ymax:
            ymax, ymin = ymin, ymax
        
        if ymin==ymax:
            y = [ymin]
        else:
            y=np.linspace(ymin, ymax, ysize)
        
        if zmin>zmax:
            zmax, zmin = zmin, zmax
        
        if zmin==zmax:
            z = [zmin]
        else:
            z=np.linspace(zmin, zmax, zsize)
         
        return Grid(np.meshgrid(x, y, z, indexing='ij'))
        
