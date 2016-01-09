import os
import nrrd
import numpy as np
import nibabel as nib
import cv2

path = '../../data'

images = []
folders = []

for subdir, dirs, files in os.walk(path):
	for file1 in files:
		if 'img' in file1:
			images.append(subdir + '/' + file1)
			folders.append(subdir)


for i in xrange(len(images)):

	# if i == 1:
	# 	break

	print '==> Iteration: ', i+1

	image, options = nrrd.read(images[i])
	blurred = cv2.GaussianBlur(image,(5,5),0)

	nrrd.write(folders[i] + '/gaussianBlurred.nrrd', image)