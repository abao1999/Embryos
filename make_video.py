'''
Takes sequence of images and creates video.
https://stackoverflow.com/questions/44947505/how-to-make-a-movie-out-of-images-in-python
'''

import cv2
import os

# Directory with all images
image_folder = 'videos_bf_sequence'
video_name = 'embryo_13_bf_sequence.mp4'

# Find list of image filenames
images = [img for img in os.listdir(image_folder) if img.endswith('.png')]
images.sort(key=lambda x: int(x.split('_')[0]))

frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape
# Framerate
fps = 5
# Specify video codec
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()