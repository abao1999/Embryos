# pix2pix

This folder contains our work on image translation, specifically converting from the bright-field to the fluorescence channel. Here is an overview of the files:
1. `pix2pix_PyTorch.ipynb` - Implementation of pix2pix in PyTorch (which will be developed into an end-to-end model with a polarization classifier)
2. `pix2pix_TensorFlow.ipynb` - Implementation of pix2pix in TensorFlow (with the standard pix2pix loss)
3. `evaluate_pix2pix.ipynb` - Notebook to restore from a TensorFlow pix2pix checkpoint, generate fluo-channel test output tied to true polarization labels, and evaluate the output using a pretrained polarization classifier.
