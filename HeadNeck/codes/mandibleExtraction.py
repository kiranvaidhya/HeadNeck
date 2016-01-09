from sklearn.feature_extraction import image
import nrrd
import os
import getopt
import sys
import numpy as np
from scipy.ndimage import zoom

path = '../data/Training'
patch_root = 'patches/'

patch_size = 11
padding = 20
validation = False

options, remainder = getopt.getopt(sys.argv[1:], 'p:v', ['patch_size=','validation',])

# "python 1_extractPatches.py -p 31 -v" for extracting validation patches of size 31
# Drop -v if you want to extract training patches which will do the data augmentation
for opt, arg in options:
    if opt in ('-p', '--patchSize'):
        patch_size = int(arg)
    elif opt in ('-v', '--validation'):
        validation = True

print 'patchSize   :', patch_size
print 'validation   :', validation

if validation == True:
	path = '../data/Validation'
	patch_root = 'validationPatches/'

images = []
truths = []
folders = []

for subdir, dirs, files in os.walk(path):
	for file1 in files:
		if 'combined' in file1:
			truths.append(subdir + '/' + file1)
		if 'normalizedImg' in file1:
			images.append(subdir + '/' + file1)
			folders.append(subdir)



for i in xrange(len(images)):

	# if i == 1:
	# 	break

	print
	print
	print '####################################################################'
	print '==> Extracting from image: ', i+1
	print '    Folder: ', folders[i]
	print '####################################################################'
	print
	print

	img, options = nrrd.read(images[i])
	truth, options = nrrd.read(truths[i])
	folder = folders[i]

	truth_pixels = []

	patches = np.zeros((1,patch_size,patch_size))
	ground_truths = np.zeros((1))

	print '==> Extracting patches..'

	labelList = [3]

	for z in labelList:

		try:

		# print '==> Extracting patches for label: ', z

			x_span, y_span, z_span = np.where(truth == z)
			truth_pixels.append(np.where(truth == z))


			start_slice = min(z_span)
			stop_slice = max(z_span)

			x_start = min(x_span) - padding
			x_stop = max(x_span) + padding + 1

			y_start = min(y_span) - padding
			y_stop = max(y_span) + padding + 1

			if ((x_stop - x_start) * (y_stop - y_start)) < patch_size * patch_size:
				x_start = x_start - 20
				x_stop = x_stop + 20
				y_start = y_start - 20
				y_stop = y_stop + 20

			for j in xrange(start_slice, stop_slice):

				image_slice = img[:,:,j]
				truth_slice = truth[:,:,j]

				patch = image.extract_patches(image_slice[x_start:x_stop,y_start:y_stop], patch_size, extraction_step = 1)
				patch = patch.reshape(patch.shape[0]*patch.shape[1],patch_size,patch_size)

				truth_patch = image.extract_patches(truth_slice[x_start:x_stop,y_start:y_stop], patch_size, extraction_step = 1)
				truth_patch = truth_patch.reshape(truth_patch.shape[0]*truth_patch.shape[1],patch_size,patch_size)

				truth_values = truth_patch[:, (patch_size - 1)/2, (patch_size -1)/2]

				patches = np.append(patches,patch,axis=0)
				ground_truths = np.append(ground_truths,truth_values,axis=0)

			patches = patches[1:patches.shape[0]]
			ground_truths = ground_truths[1:ground_truths.shape[0]]

		except:
			print '==> Error in label: ', z

	patches = patches[np.where(ground_truths==3)]
	ground_truths = ground_truths[np.where(ground_truths==3)]
	ground_truths[np.where(ground_truths==3)] = 1

	print 'Mandible:				1	', np.sum((ground_truths==1).astype(int))

	np.save(patch_root + 'patches_' + str(i) +'.npy', patches)
	np.save(patch_root + 'truths_' + str(i) + '.npy', ground_truths)