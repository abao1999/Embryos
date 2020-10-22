import os
import h5py
import numpy as np
from notebooks import utils

bf_data_path = 'data/video_bf_data'
fluo_data_path = 'data/video_fluo_data'

bf_processed_path = 'processed/bf_data'
fluo_processed_path = 'processed/fluo_data'

funcs = [('middle', utils.get_middle_z), ('max', utils.max_across_z),
         ('min', utils.min_across_z), ('avg', utils.avg_across_z)]

embryo_idx_arr = [47, 49, 50, 52, 53]

def load_bf_video(embryo_idx):
    bf = h5py.File(os.path.join(bf_data_path,'embryo_'+str(embryo_idx)+'.mat'))
    arrays = {}
    for k, v in bf.items():
        arrays[k] = np.array(v)
    bf_video = arrays['data']
    pol_state = arrays['anno']
    return bf_video, pol_state

def load_fluo_video(embryo_idx):
    fluo = h5py.File(os.path.join(fluo_data_path,'embryo_'+str(embryo_idx)+'.mat'))
    arrays = {}
    for k, v in fluo.items():
        arrays[k] = np.array(v)
    fluo_video = arrays['data']
    pol_state = arrays['anno']
    return fluo_video, pol_state

def process_and_save(embryo_idx, video, pol_state, top_path):
    '''
    Applies each type of preprocessing to the video for idx and saves as npy.
    '''
    for sub_dir, f in funcs:
        result = f(video)
        np.save(f'{top_path}/{sub_dir}/embryo_{embryo_idx}', result)

if __name__ == '__main__':
    # Make all directories that do not yet exist.
    if not os.path.isdir('processed'):
        os.mkdir('processed')
    for top_path in [bf_processed_path, fluo_processed_path]:
        if not os.path.isdir(top_path):
            os.mkdir(top_path)
        for sub_path, _ in funcs:
            full_path = f'{top_path}/{sub_path}'
            if not os.path.isdir(full_path):
                os.mkdir(full_path)

    # Process each idx with each method in funcs
    # Note: polarization data is currently not used
    for idx in embryo_idx_arr:
        bf_video, pol_state = load_bf_video(idx)
        process_and_save(idx, bf_video, pol_state, bf_processed_path)
        fluo_video, pol_state = load_fluo_video(idx)
        process_and_save(idx, fluo_video, pol_state, fluo_processed_path)
