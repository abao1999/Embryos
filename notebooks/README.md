# Notebooks

Here is an updated list of notebooks for reproducible results:
1. pretrained/classify_pretrained.ipynb - Implementation of ImageNet-pretrained classifier with GluonCV
2. pretrained/visualize_model_features.ipynb - Analyze GluonCV classifier (confusion matrix, class activation maps, mistakes)
3. pix2pix/pix2pix_TensorFlow.ipynb - 

Here is a list of notebooks that are not intended for reproducible runs:
1. *mnist_cnn.ipynb - online tutorial for MNIST classification with CNN
2. *classify_fluo.ipynb - adaptation of mnist_cnn.ipynb, using local data
3. classify_fluo_pretrained_torch.ipynb - transfer learning on pretrained PyTorch model
4. *classify_fluo_CNN.ipynb - combo of classify_fluo.ipynb / classify_fluo_pretrained.ipynb, using original cs101 drive
5. *classify_fluo_CNN_2.ipynb - similar to classify_fluo_CNN.ipynb, using new shared drive
6. explore.ipynb - adaptation of original starter notebook provided by Cheng
7. *classify_fluo_CNN_single_output.ipynb - uses 1 output dim instead of 10
8. *classify_fluo_CNN_torch.ipynb - PyTorch reimplementation of previous CNN
9. classify_fluo_CNN_torch_drive.ipynb - same as 8 but with new shared drive
10. classify_fluo_CNN_3D_torch_drive.ipynb - using 3D instead of 2D
11. visualize_model_features_torch.ipynb - for visualizations for PyTorch classifier

Notebooks with * were moved to outdated/

Models folder contains architectures of some models used:
1. 2dcnn1.py: Simple CNN
2. convgru.py: Convolutional GRU, replacing the pointwise matrix operations within the GRU cell's gates with convolutional operations

## Utils

The file utils.py contains many helpful functions that our pipelines and models rely on. These functions include normalization functions, functions to select embryo slices, aggregation functions, train-test split helpers, and functions to convert and save the data as various file formats.
