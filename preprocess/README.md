# Preprocess

This directory stores various scripts to convert the original dataset (as ".mat" data or Google Drive images) into NumPy arrays for individual embryos. It also contains code for initial visualization/exploration of the embryo samples.

Running the `preprocess/` scripts locally (i.e., not on Colab or HPC) requires a top-level `data/` directory and will create a top-level `processed/` directory (which are excluded from the repository via the `.gitignore` file). You should save embryos to the `data/` folder with naming conventions `data/video_bf_data/embryo_{embryo_idx}.mat` and `data/video_fluo_data/embryo_{embryo_idx}.mat` for bright-field and fluorescence data, respectively.

Here is an overview of the files for data conversion/preparation:
1. `preprocess_mat_to_npy.py` - This script converts the `embryo_{embryo_idx}.mat` files into NumPy arrays for bright-field, fluorscence, and polarization data. To run, modify the script to hard-code a list of embryo indices to process (as the variable `embryo_idx_arr`). Here is an example: `embryo_idx_arr = [47, 49, 50, 52, 53]`
2. `preprocess_mat_to_npy_2.py` - This script is the same as `preprocess_mat_to_npy.py` but it inputs the array of embryo indices as a command line argument. Here is an example: `python3 preprocess_mat_to_npy_2.py --embryo-indices 47,49,50,52,53`
3. `preprocess_drive.ipynb` - This notebook is for converting the Google Drive representation of embryo images into NumPy arrays (and saving them in Google Drive). It is comparable to the Python scripts, but only runs in the Colab environment.

Here is an overview of the files for data visualization/exploration:
1. `make_video.py` - Creates a video from a folder of images (e.g., `videos_bf_sequence/`)
2. `vis.py` - Visualizes slices of embryo data (using local data).
3. `explore.ipynb` - Notebook to combine some of the initial data analysis (including correlation analysis between channels).
