import os
import nrrd
import numpy as np
import nibabel as nib

path = '../../data'

images = []
folders = []

for subdir, dirs, files in os.walk(path):
	for file1 in files:
		if 'img' in file1:
			images.append(subdir + '/' + file1)
			folders.append(subdir)

means = []
stds = []

for i in xrange(len(images)):

	print '==> Iteration: ', i+1

	image, options = nrrd.read(images[i])
	means.append(np.mean(image))
	stds.append(np.std(image))

print '==> Calculating Global Mean and Global STD'

globalMean = np.mean(means)
globalStd = np.mean(stds)

print '==> Normalizing Images'

for i in xrange(len(images)):

	# if i == 1:
	# 	break

	print '==> Iteration: ', i+1

	image, options = nrrd.read(images[i])
	image = (image - globalMean)/globalStd

	# affine=[[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
	# img = nib.Nifti1Image(image, affine)
	# img.set_data_dtype(np.int32)
	# nib.save(img,folders[i] + '/normalizedImg.nii')

	nrrd.write(folders[i] + '/normalizedImg.nrrd', image)