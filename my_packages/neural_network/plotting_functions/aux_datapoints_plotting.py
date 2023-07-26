import numpy as np
import matplotlib.pyplot as plt

from my_packages.classes.field_classes import Scan


def plotting_func(Ez, Hx, Hy, x, y, mask, ax=None, FIGSIZE=(15,3)):
    zr0x = x.v[mask[0] == 1]
    zr0y = y.v[mask[0] == 1]

    xr0x = x.v[mask[1] == 1]
    xr0y = y.v[mask[1] == 1]

    yr0x = x.v[mask[2] == 1]
    yr0y = y.v[mask[2] == 1]

    if ax is None:
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize=FIGSIZE, constrained_layout=True)
    else:
        ax1, ax2, ax3 = ax

    Ez.plot(ax=ax1, title="Ez")
    ax1.scatter(zr0x, zr0y, c="w", marker="s", edgecolor="k", label="z oriented")
    ax1.scatter(xr0x, xr0y, c="r", marker=">", edgecolor="k", label="x oriented")
    ax1.scatter(yr0x, yr0y, c="k", marker="^", label="y oriented")
    
    Hx.plot(ax=ax2, title="Hx")
    ax2.scatter(zr0x, zr0y, c="w", marker="s", edgecolor="k", label="z oriented")
    ax2.scatter(xr0x, xr0y, c="r", marker=">", edgecolor="k", label="x oriented")
    ax2.scatter(yr0x, yr0y, c="k", marker="^", label="y oriented")
    
    
    Hy.plot(ax=ax3, title="Hy")
    ax3.scatter(zr0x, zr0y, c="w", marker="s", edgecolor="k", label="z oriented")
    ax3.scatter(xr0x, xr0y, c="r", marker=">", edgecolor="k", label="x oriented")
    ax3.scatter(yr0x, yr0y, c="k", marker="^", label="y oriented")


    ax1.legend(loc="upper center", bbox_to_anchor=(-0.5, 0.95), frameon=True,
                handlelength=2.5, handletextpad=0.5, ncol=1)

    return fig, (ax1, ax2, ax3)

def H_plotting_func(Hx, Hy, x, y, mask, ax=None, FIGSIZE=(15,3)):
        xr0x = x.v[mask[0] == 1]
        xr0y = y.v[mask[0] == 1]

        yr0x = x.v[mask[1] == 1]
        yr0y = y.v[mask[1] == 1]

        if ax is None:
            fig, (ax1, ax2) = plt.subplots(1,2, figsize=FIGSIZE, constrained_layout=True)
        else:
            ax1, ax2 = ax

        Hx.plot(ax=ax1, title="Hx")
        ax1.scatter(xr0x, xr0y, c="r", marker=">", edgecolor="k", label="x oriented")
        ax1.scatter(yr0x, yr0y, c="k", marker="^", label="y oriented")

        Hy.plot(ax=ax2, title="Hy")
        ax2.scatter(xr0x, xr0y, c="r", marker=">", edgecolor="k", label="x oriented")
        ax2.scatter(yr0x, yr0y, c="k", marker="^", label="y oriented")

        ax1.legend(loc="upper center", bbox_to_anchor=(-0.5, 0.95), frameon=True,
                handlelength=2.5, handletextpad=0.5, ncol=1)
        

def E_plotting_func(Ez: Scan, x, y, mask, ax=None, FIGSIZE=(15,3)):
        zr0x = x.v[mask[0] == 1]
        zr0y = y.v[mask[0] == 1]

        if ax is None:
            fig, ax1 = plt.subplots(1,1, figsize=FIGSIZE, constrained_layout=True)
        else:
            ax1 = ax
        
        Ez.plot(ax=ax1, title="Ez")
        ax1.scatter(zr0x, zr0y, c="w", marker="s", edgecolor="k", label="z oriented")

        ax1.legend(loc="upper center", bbox_to_anchor=(-0.5, 0.95), frameon=True,
                    handlelength=2.5, handletextpad=0.5, ncol=1)    


def input_data_to_Scan(data, grid, freq=1e9):
    if np.ndim(data) == 3:
        data = np.expand_dims(data, axis=1)
    n_probe_heights = data.shape[1]

    fields_p_height = [_input_data_to_Scan_single_layer(data[:,ii, ...], grid=grid[..., ii], freq=freq) for ii in range(n_probe_heights)]
    Ez, Hx, Hy = zip(*fields_p_height)

    for x in Ez, Hx, Hy:
        if any(isinstance(t, type(None)) for t in x):
            x=None

    return Ez, Hx, Hy

def _input_data_to_Scan_single_layer(data, grid, freq=1e9):
    if len(data)==3:
        Ez = Scan(data[0], grid=grid, freq=freq, axis="z", component="z", field_type="E")
        Hx = Scan(data[1], grid=grid, freq=freq, axis="z", component="x", field_type="H")
        Hy = Scan(data[2], grid=grid, freq=freq, axis="z", component="y", field_type="H")
        return Ez, Hx, Hy
    elif len(data)==2:
        Hx = Scan(data[0], grid=grid, freq=freq, axis="z", component="x", field_type="H")
        Hy = Scan(data[1], grid=grid, freq=freq, axis="z", component="y", field_type="H")
        return None, Hx, Hy
    elif len(data)==1:
        Ez = Scan(data[0], grid=grid, freq=freq, axis="z", component="z", field_type="E")
        return Ez, None, None