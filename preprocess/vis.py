# -*- coding: utf-8 -*-
"""
Created on Sat Aug  8 15:04:01 2020

@author: https://www.datacamp.com/community/tutorials/matplotlib-3d-volumetric-data#Plotting
Other visualization sources:
    https://plotly.com/python/3d-volume-plots/
    https://plotly.com/python/visualizing-mri-volume-slices/
    http://hyperspy.org/hyperspy-doc/current/user_guide/intro.html
"""

import matplotlib.pyplot as plt
from skimage import io

def remove_keymap_conflicts(new_keys_set):
    for prop in plt.rcParams:
        if prop.startswith('keymap.'):
            keys = plt.rcParams[prop]
            remove_list = set(keys) & new_keys_set
            for key in remove_list:
                keys.remove(key)

def multi_slice_viewer(volume):
    remove_keymap_conflicts({'j', 'k'})
    fig, ax = plt.subplots()
    ax.volume = volume
    ax.index = volume.shape[0] // 2
    ax.imshow(volume[ax.index])
    fig.canvas.mpl_connect('key_press_event', process_key)
    plt.show()

def process_key(event):
    fig = event.canvas.figure
    ax = fig.axes[0]
    if event.key == 'j':
        previous_slice(ax)
    elif event.key == 'k':
        next_slice(ax)
    fig.canvas.draw()

def previous_slice(ax):
    volume = ax.volume
    ax.index = (ax.index - 1) % volume.shape[0]  # wrap around using %
    ax.images[0].set_array(volume[ax.index])

def next_slice(ax):
    volume = ax.volume
    ax.index = (ax.index + 1) % volume.shape[0]
    ax.images[0].set_array(volume[ax.index])

if __name__ == "__main__":
    struct_arr = io.imread("https://s3.amazonaws.com/assets.datacamp.com/blog_assets/attention-mri.tif")
    multi_slice_viewer(struct_arr.T)
