import numpy as np

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

def max_across_z(img, normalize=False):
	""" Returns a new image where each pixel
	intensity is the maximum for that pixel across
	all images in the z-stack. 
	"""

	if normalize:
		img = normalize(img)

	return np.amax(img, axis=0)

def min_across_z(img, normalize=False):
	""" Returns a new image where each pixel
	intensity is the minimum for that pixel across
	all images in the z-stack. 
	"""

	if normalize:
		img = normalize(img)

	return np.amin(img, axis=0)

def avg_across_z(img, normalize=False):
	""" Returns a new image where each pixel
	intensity is the average for that pixel across 
	all images in the z-stack. 
	"""

	if normalize:
		img = normalize(img)

	return np.mean(img, axis=0)





