# Embryos
This is the code repository for the CS101 Embryo Polarization project (fall 2020).

## Abstract

The polarization of the apical cell domains of embryos at the 8-cell stage marks an important milestone in development. Current techniques to determine the presence of this polar domain require invasive fluorescence imaging and are not always appropriate. We explore the potential of applying machine learning and computer vision techniques to predict polarization in these embryos from non-invasive bright field images only. Furthermore, we explore artificially generating an accurate fluorescent image from a bright field one using the pix2pix model.

## Repository Organization

The `notebooks/` directory stores Jupyter notebooks which generated our core results. This directory contains internal documentation on specific models and methods.

The `preprocess/` directory stores various scripts to convert the original dataset (as ".mat" data or Google Drive images) into NumPy arrays for individual embryos. It also contains code for initial visualization/exploration of the embryo samples.

Running the `preprocess/` scripts locally (i.e., not on Colab or HPC) requires the creation of `data/` and `processed/` directories (which are excluded from the repository via the `.gitignore` file). In general, we recommend running the provided models on Colab due to potentially long runtimes and/or high memory requirements.
