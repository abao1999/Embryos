import numpy as np
import os
import pandas as pd
from PIL import Image

# The following constants are different idx to train on:
PARTIAL_IDX = [1, 3, 12, 13, 16, 18, 19, 24, 39, 40, 42, 45, 46, 47, 49, 50, 52]
FULL_IDX = PARTIAL_IDX + [76, 78, 79, 80, 81, 82, 83, 84, 85, 87, 88, 92, 93, 94, 95, 96, 97, 99, 100, 101, 102]
CLEAN_IDX = [3, 12, 13, 16, 18, 19, 23, 24, 25, 33, 39, 42, 46, 47, 76, 78, 79,
             80, 81, 85, 87, 88, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102]
CLEAN_IDX_FULL = CLEAN_IDX + [53, 77, 103, 104, 105, 106, 107, 108, 109, 110, 111,
                              112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                              122, 124, 125, 126, 127, 128, 129, 130, 131, 132,
                              133, 134, 135, 137, 138, 139, 140, 141, 142, 143,
                              144, 145, 147, 148, 149, 150, 151 , 152, 153, 155]

# The following functions are used in preprocessing raw data:

def get_z_slice(z, img):
    assert len(img.shape) == 4
    return img[z, :, :, :]

def get_img_at_t(t, img):
    assert len(img.shape) == 4
    return img[:, :, :, t]

def normalize(img):
    """ Normalizes pixel values across all images in img
    to range 0-1.
    """
    assert len(img.shape) == 4

    temp = img - np.min(img)
    if np.max(temp) != 0:
        b = temp / np.max(temp)
    else:
        b = temp
    return b

def middle_z(img):
    assert len(img.shape) == 4

    zdim, xdim, ydim, tdim = img.shape
    result = np.empty(shape=(1, xdim, ydim, tdim))
    result[0] = get_z_slice(int(img.shape[0] / 2), img)
    return result

def max_across_z(img, normalize=False):
    """ Returns a new image where each pixel
    intensity is the maximum for that pixel across
    all images in the z-stack.
    """

    if normalize:
        img = normalize(img)

    zdim, xdim, ydim, tdim = img.shape
    result = np.empty(shape=(1, xdim, ydim, tdim))

    result[0] = np.amax(img, axis=0)
    return result

def min_across_z(img, normalize=False):
    """ Returns a new image where each pixel
    intensity is the minimum for that pixel across
    all images in the z-stack.
    """

    if normalize:
        img = normalize(img)

    zdim, xdim, ydim, tdim = img.shape
    result = np.empty(shape=(1, xdim, ydim, tdim))

    result[0] = np.amin(img, axis=0)
    return result

def avg_across_z(img, normalize=False):
    """ Returns a new image where each pixel
    intensity is the average for that pixel across
    all images in the z-stack.
    """

    if normalize:
        img = normalize(img)

    zdim, xdim, ydim, tdim = img.shape
    result = np.empty(shape=(1, xdim, ydim, tdim))

    result[0] = np.mean(img, axis=0)
    return result

# The following functions are used to prepare training/testing sets on drive

# def prepare_drive():
#     """ Runs system command to ensure shared drive
#     mounts properly.
#     """
#     # Reference: https://github.com/googlecolab/colabtools/issues/1494
#     cmd = "!sed -i -e 's/enforce_single_parent:true/enforce_single_parent:true,metadata_cache_reset_counter:4/' /usr/local/lib/python3.6/dist-packages/google/colab/drive.py"
#     os.system(cmd)

def split_train_test_val(home_path, embryo_inds):
    """ Splits a set of embryos into train/test/val:
    home_path: path to shared drive folder
    embryo_inds: list of embryos indices to use
    """
    # Load info about videos
    video_time_info = pd.read_excel(f'{home_path}/embryo_info_CS101.xlsx', index_col=0, header=0, na_values=['NaN','NAN'], usecols=['embryo_index', 'first_anno_pol_time', 't_num'])
    video_time_info.dropna(inplace=True, subset=['first_anno_pol_time'])
    print(video_time_info.loc[embryo_inds])

    p = np.random.permutation(len(embryo_inds))
    p_embryo = [embryo_inds[i] for i in p]
    t_num = list(video_time_info.loc[embryo_inds, 't_num'])
    t_num_random = list(video_time_info.loc[p_embryo, 't_num'])

    instance_cum_random = np.cumsum(t_num_random)
    test_split_point = instance_cum_random[-1]*0.83
    temp = abs(instance_cum_random-test_split_point)
    test_idx = np.argmin(temp)

    val_split_point = instance_cum_random[-1]*0.7
    temp = abs(instance_cum_random-val_split_point)
    val_idx = np.argmin(temp)

    train_embryos = p_embryo[:val_idx]
    val_embryos = p_embryo[val_idx:test_idx]
    test_embryos = p_embryo[test_idx:]
    print(train_embryos)
    print(val_embryos)
    print(test_embryos)

    return video_time_info, train_embryos, val_embryos, test_embryos

def within_window(embryo_idx, t, window, video_time_info):
    ''' Returns if an embryo timestep is close to the 1st polarized time
    embryo_idx: index of current embryo (value from 'embryo_index' col)
    t: timestep of current
    window: number of t steps from first polarized index to ignore
    video_time_info: dataframe with labeling/timing information per embryo
    '''
    first_pol_idx = video_time_info.loc[embryo_idx, 'first_anno_pol_time'] - 1
    return window and abs(first_pol_idx - t) <= window

def get_max_pixel(embryos, data_path):
    ''' Obtains the maximum pixel value across a set of embryos
    embryos: subset of p_embryo... train, val, test
    data_path: path from which to load processed np embryo data
    '''
    max_per_embryo = []
    for i in range(len(embryos)):
        embryo_idx = embryos[i]
        embryo_path = f'{data_path}/embryo{embryo_idx}.npy'
        try:
            embryo = np.load(embryo_path)
        except FileNotFoundError:
            continue
        max_per_embryo.append(np.max(embryo))
    return max(max_per_embryo)

def save_nps_as_png(embryos, save_path, specs, window=None, normalize='per_embryo', dim=2, pol_subdir=True):
    ''' Save dataset in image format, sorted by polarization state
    embryos: subset of p_embryo... train, val, test
    specs = (data_path, pol_path, video_time_info)
    save_path: path to save png to... data_path + {'train', 'val', 'test'}
    specs: tuple with data_path (str: processed np embryo data),
                      pol_path (str: embryo polarization labels),
                      video_time_info (df: info per embryo)
    window: number of t steps from first polarized index to ignore
    normalize: type of normalization to apply (per_embryo, per_timestep, #)
    dim: 2 or 3 (using 2d representation of z-stack or 3d selection of slices)
    pol_subdir: whether to save images to "{save_path}/{pol}/" or "{save_path}/"
    '''
    data_path, pol_path, video_time_info = specs
    for i in range(len(embryos)):
        embryo_idx = embryos[i]
        embryo_path = f'{data_path}/embryo{embryo_idx}.npy'
        embryo_pol_path = f'{pol_path}/embryo{embryo_idx}.npy'
        try:
            embryo = np.load(embryo_path)
        except FileNotFoundError:
            print("File not found", embryo_path)
            continue
        embryo_pol = np.squeeze(np.load(embryo_pol_path)).astype(int)

        # pre-normalization steps
        # embryo is currently np.float32, but for 3D input
        # this conversion step exceeds Colab RAM usage
        if dim == 2:
            embryo = embryo.astype(np.float64)

        # normalize the data to 0 - 1
        if normalize == 'per_embryo': # max val over the full embryo
            embryo /= np.max(embryo)
        elif normalize == 'per_timestep': # max val over each timestep
            embryo /= np.max(embryo, axis=(0,1))
        elif type(normalize) is not str: # a fixed numerical factor
            embryo /= normalize
        embryo = 255 * embryo # Now scale by 255
        embryo = embryo.astype(np.uint8)

        # Scale from (1,x,y,t) -> (x,y,t) if using 2D image input
        if dim == 2:
            if len(embryo.shape) == 4:
                embryo = embryo[0]

        print(embryo_idx, np.shape(embryo)[-1])
        for t in range(np.shape(embryo)[-1]):
            if within_window(embryo_idx, t, window, video_time_info):
                print(f'skipping embryo {embryo_idx} step {t}')
                continue
            pol = embryo_pol[t]

            # Set the specific sub-path of save_path to save the img
            save_dir = f'{save_path}/{pol}' if pol_subdir else save_path

            # Add a suffix for the pix2pix model (polarization label)
            pol_suffix = f'_{pol}' if not pol_subdir else ''

            # Save as images if using 2D image as tstep input
            if dim == 2:
                img = Image.fromarray(embryo[:,:,t], 'L')
                img_path = f'{save_dir}/embryo_{embryo_idx}_{t}{pol_suffix}.png'
                img.save(img_path)

            # Save as npy if using 3D slices as tstep input
            if dim == 3:
                mid = int(embryo.shape[0] / 2)
                slices = embryo[mid-1:mid+2,:,:,t]
                slices_path = f'{save_dir}/embryo_{embryo_idx}_{t}{pol_suffix}.npy'
                np.save(slices_path, slices)

                # memory optimization
                slices = None

        # memory optimization
        embryo = None

def save_nps_subset_z(embryos, save_path, specs, window=None, normalize='per_embryo', mode='middle', n_slices=3, pol_subdir=True):
    ''' Save dataset in image format, sorted by polarization state
    embryos: subset of p_embryo... train, val, test
    specs = (data_path, pol_path, video_time_info)
    save_path: path to save png to... data_path + {'train', 'val', 'test'}
    specs: tuple with data_path (str: processed np embryo data),
                      pol_path (str: embryo polarization labels),
                      video_time_info (df: info per embryo)
    window: number of t steps from first polarized index to ignore
    normalize: type of normalization to apply (per_embryo, per_timestep, #)
    mode: min, middle, max... where to start z slices (if middle, get z slices around middle)
    n_slices: number of z slices to get
    pol_subdir: whether to save images to "{save_path}/{pol}/" or "{save_path}/"
    '''
    if mode not in ['min', 'middle', 'max'] or n_slices < 1:
        print("Please enter something valid")
        return
    data_path, pol_path, video_time_info = specs
    for i in range(len(embryos)):
        embryo_idx = embryos[i]
        embryo_path = f'{data_path}/embryo{embryo_idx}.npy'
        embryo_pol_path = f'{pol_path}/embryo{embryo_idx}.npy'
        try:
            embryo = np.load(embryo_path)
        except FileNotFoundError:
            print("File not found", embryo_path)
            continue
        embryo_pol = np.squeeze(np.load(embryo_pol_path)).astype(int)

        # pre-normalization steps
        # embryo is currently np.float32, but for 3D input
        # this conversion step exceeds Colab RAM usage
        embryo = embryo.astype(np.float64)

        # normalize the data to 0 - 1
        if normalize == 'per_embryo': # max val over the full embryo
            embryo /= np.max(embryo)
        elif normalize == 'per_timestep': # max val over each timestep
            embryo /= np.max(embryo, axis=(0,1))
        elif type(normalize) is not str: # a fixed numerical factor
            embryo /= normalize
        embryo = 255 * embryo # Now scale by 255
        embryo = embryo.astype(np.uint8)

        print(embryo_idx, np.shape(embryo)[-1])
        for t in range(np.shape(embryo)[-1]):
            if within_window(embryo_idx, t, window, video_time_info):
                print(f'skipping embryo {embryo_idx} step {t}')
                continue
            pol = embryo_pol[t]

            # Set the specific sub-path of save_path to save the img
            save_dir = f'{save_path}/{pol}' if pol_subdir else save_path

            # Add a suffix for the pix2pix model (polarization label)
            pol_suffix = f'_{pol}' if not pol_subdir else ''
            
            # save as npy
            if mode == 'middle':
                mid = embryo.shape[0]//2
                try:
                    slices = embryo[mid-(n_slices//2):mid+(n_slices - n_slices//2),:,:,t]
                except IndexError:
                    print("z slices went out of bounds")
                    continue
                    
            elif mode == 'min':
                try:
                    slices = embryo[0:n_slices,:,:,t]
                except IndexError:
                    print("z slices went out of bounds")
                    continue
                    
            elif mode == 'max':
                try:
                    slices = embryo[-1:-1-n_slices:-1,:,:,t]
                except IndexError:
                    print("z slices went out of bounds")
                    continue
                    
            else:
                print("something unexpected happened")
            
            slices_path = f'{save_dir}/embryo_{embryo_idx}_z{mode}{n_slices}_t{t}_pol{pol_suffix}.npy'
            np.save(slices_path, slices)

            # memory optimization
            slices = None

        # memory optimization
        embryo = None

