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

#%%
# read a set of embryo data
if onedrive_data:
    bf_data_path = 'D:/New_OneDrive/OneDrive - California Institute of Technology/7. AI Embryo/4. Data/video_bf_data'
    fluo_data_path = 'D:/New_OneDrive/OneDrive - California Institute of Technology/7. AI Embryo/4. Data/video_fluo_data'
else:
    bf_data_path = 'data/video_bf_data'
    fluo_data_path = 'data/video_fluo_data'

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

#%%
# correlation
