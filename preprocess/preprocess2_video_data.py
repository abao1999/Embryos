# -*- coding: utf-8 -*-
"""
Created on Mon Oct 12 10:16:38 2020

@author: hgdsh
"""
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from vis import multi_slice_viewer
from scipy.ndimage import correlate, maximum_filter, minimum_filter

# Specify whether to use a OneDrive path or the local data/ folder
onedrive_data = False

def normalize_img(a):
    # normalize each channel of 2D/3D image to [0,1]
    temp = a - np.min(a)
    if np.max(temp) != 0:
        b = temp/np.max(temp)
    else:
        b = temp
    return b

def visualize(images, masks, titles):
    """PLot images in one row and masks in one row."""
    num_img = len(images)
    fig, axes = plt.subplots(2, num_img, figsize=(16, 5), sharex=True, sharey=True)
    ax = axes.ravel()
    for k in range(num_img):
        ax[k].imshow(normalize_img(np.squeeze(images[k,:,:,])))
        ax[k].axis('off')
        ax[k].set_title(titles[k]+' bf image')

        ax[k+num_img].imshow(normalize_img(np.squeeze(masks[k,:,:])))
        ax[k+num_img].axis('off')
        ax[k+num_img].set_title(titles[k]+' fluo. image')
    plt.show()

def save_all_images(images, name, dir_name):
    for t in range(total_time):
        fn = '{}/{}_{}.png'.format(dir_name, t, name)
        plt.imsave(fn, normalize_img(np.squeeze(images[t,:,:])))

def bf_fluo_correlate(embryo_idx, bf_image, fluo_image, normalized=False, smoothed=None, dim=3):
    """Plot correlation between bf/fluo images, pixel-by-pixel or smoothed in tiles."""
    norm_str = ('N' if normalized else 'Unn') + 'ormalized'
    smooth_str = 'unsmoothed'
    if normalized:
        bf_image = normalize_img(bf_image)
        fluo_image = normalize_img(fluo_image)
    if smoothed:
        if smoothed == 'avg':
            weights = [[1/(dim**2)]*dim]*dim
            bf_image = correlate(bf_image, weights)
            fluo_image = correlate(fluo_image, weights)
        elif smoothed == 'max':
            bf_image = maximum_filter(bf_image, size=dim)
            fluo_image = maximum_filter(fluo_image, size=dim)
        elif smoothed == 'min':
            bf_image = minimum_filter(bf_image, size=dim)
            fluo_image = minimum_filter(fluo_image, size=dim)
        else:
            raise Exception('smoothed can only be avg, max, or min')
        smooth_str = 'smoothed by ' + smoothed
    plt.scatter(bf_image.flatten(), fluo_image.flatten())
    plt.xlabel(f"{norm_str} bf frame val")
    plt.ylabel(f"{norm_str} fluo frame val")
    plt.title(f"Correlation plot for {smooth_str} embryo {embryo_idx}")
    plt.show()


#%%
# read a set of embryo data
if onedrive_data:
    bf_data_path = 'D:/New_OneDrive/OneDrive - California Institute of Technology/7. AI Embryo/4. Data/video_bf_data'
    fluo_data_path = 'D:/New_OneDrive/OneDrive - California Institute of Technology/7. AI Embryo/4. Data/video_fluo_data'
else:
    bf_data_path = '../data/video_bf_data'
    fluo_data_path = '../data/video_fluo_data'

embryo_idx = 1
bf = h5py.File(os.path.join(bf_data_path,'embryo_'+str(embryo_idx)+'.mat'))
arrays = {}
for k, v in bf.items():
    arrays[k] = np.array(v)
bf_video = arrays['data']
pol_state = arrays['anno']

fluo = h5py.File(os.path.join(fluo_data_path,'embryo_'+str(embryo_idx)+'.mat'))
arrays = {}
for k, v in fluo.items():
    arrays[k] = np.array(v)
fluo_video = arrays['data']
pol_state = arrays['anno']

#%%
# vis polarized vs unpolarized
num_nonpol = np.sum(pol_state==0)
num_pol = np.sum(pol_state==1)
print('In embryo %d, %d nonpolarized frames and %d polarized frames' % (embryo_idx,num_nonpol,num_pol))
nonpol_idx = np.random.choice(np.where(pol_state==0)[1],1)
pol_idx = np.random.choice(np.where(pol_state==1)[1],1)
bf_images = np.squeeze(bf_video[7,:,:,[nonpol_idx,pol_idx]])
fluo_images = np.squeeze(fluo_video[7,:,:,[nonpol_idx,pol_idx]])
titles = ['Unpol','Pol']
visualize(bf_images, fluo_images, titles)

#%%
# view a volume
multi_slice_viewer(np.squeeze(fluo_video[:,:,:,0]))

# view a sequence
sequence = np.squeeze(fluo_video[7,:,:,:])
sequence = np.swapaxes(sequence,0,2)
multi_slice_viewer(sequence)
# Uncomment to save images from sequence
# save_all_images(sequence, 'embryo_{}_fluo_sequence'.format(embryo_idx), 'fluo_sequence')

#%%
# correlation
bf_fluo_correlate(embryo_idx, bf_images[1], fluo_images[1])
bf_fluo_correlate(embryo_idx, bf_images[1], fluo_images[1], smoothed='avg')
bf_fluo_correlate(embryo_idx, bf_images[1], fluo_images[1], smoothed='min')
bf_fluo_correlate(embryo_idx, bf_images[1], fluo_images[1], smoothed='max')
