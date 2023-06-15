
from typing import Callable

import numpy as np
import matplotlib.pyplot as plt
from moviepy.video.io.bindings import mplfig_to_npimage
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from functools import partial



def plot_working_scan(morph_model, ii, dpi=100, figsize=(7,4), **kwargs):
    scatter_kwargs = {"color":"k", "s":15}
    scatter_kwargs.update(kwargs)

    mask_to_plot = "total_mask"
    h = morph_model.history[ii]

    fig, ax = plt.subplots(figsize = figsize, dpi=dpi)
    plt_objects = h["working_scans"].plot(ax=ax, alpha=0.8)
    plt_objects["cbar"].remove()
    my_alphas = np.ones(h["binarized_scan"].scan.shape)*0.6
    my_alphas[h["binarized_scan"].scan.T == 0] = 0
    h["binarized_scan"].plot(ax=ax, alpha=my_alphas, cmap="Oranges")
    h.get(mask_to_plot).scatter_plot(ax = ax, **scatter_kwargs)
    title = ax.get_title()
    ax.set_title(title + " Iteration {}".format(ii))
    return fig
    


def turn_to_animation(func: Callable):
    def wrapper(morph_model, animation_path: str, fps: int = 10, *args, **kwargs):
        
        # Create a function that returns a frame
        def makeframe(ii):
            fig = func(morph_model, ii, *args, **kwargs)
            image = mplfig_to_npimage(fig)
            plt.close(fig)
            return image


        # Create a list of frames for the GIF
        frames = [makeframe(ii) for ii in range(len(morph_model.history))]
        end_frames = [frames[-1]]*fps
        frames.extend(end_frames)

        # Create the GIF clip
        clip = ImageSequenceClip(frames, fps=fps)

        # Write the GIF clip to a file
        clip.write_gif(animation_path, fps=fps)
    return wrapper