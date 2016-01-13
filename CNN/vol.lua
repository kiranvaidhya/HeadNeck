py = require('fb.python')
require 'cudnn'
require 'optim'

py.exec([=[
from sklearn.feature_extraction import image
import nrrd
import os
import getopt
import sys
import numpy as np
from scipy.ndimage import zoom

path = '../data/Training'
sliceRoot = 'slices/training/'

validation = False


path = '../HeadNeck/data/Testing'
sliceRoot = 'slices/Testing/'

images = []
truths = []
folders = []

for subdir, dirs, files in os.walk(path):
	for file1 in files:
		if 'Mandible' in file1:
			truths.append(subdir + '/' + file1)
		if 'normalizedImg' in file1:
			images.append(subdir + '/' + file1)
			folders.append(subdir)

slices = np.zeros((1,200,200))
ground_truths = np.zeros((1))

sizes = []
nslices = 0

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
	try:
		truth, options = nrrd.read(folders[i]+'/structures/Mandible.nrrd')
	except:
		continue
	folder = folders[i]

	

	for j in xrange(img.shape[2]):
		imgSlice = img[156:356,156:356,j]
		truthSlice = truth[:,:,j]
		imgSlice = imgSlice.reshape(1,200,200)
		slices = np.append(slices,imgSlice,axis=0)
		ground_truths = np.append(ground_truths,np.unique(truthSlice).shape[0])

	nslices = nslices + img.shape[2]
	sizes.append(nslices)
	
	print 'Mandible: ', np.sum((ground_truths==2).astype(int))

	

slices = slices[1:slices.shape[0]]
ground_truths = ground_truths[1:ground_truths.shape[0]]
	]=])

slices = py.eval('slices')
slices = slices:reshape(slices:size(1),1,200,200)
truths = py.eval('ground_truths')

model = torch.load('results/model.net')
model:evaluate()
y = torch.zeros(slices:size(1)):cuda()
outputs = torch.zeros(slices:size(1),2):cuda()
for i = 1,slices:size(1) do
	tmp1, tmp2 = model:forward(slices[i]:cuda()):max(1)
	outputs[i] = model:forward(slices[i]:cuda())
	y[i] = tmp2
end

classes = {'1','2'}

confusion = optim.ConfusionMatrix(classes)

for i = 1,slices:size(1) do
	confusion:add(outputs[i],truths[i])
end

print(confusion)

sizes = py.eval('sizes')

j = 1
for i = 1,table.getn(sizes) do
	tmp = y[{{j,sizes[i]}}]
	f = io.open('prem'..tostring(i)..'txt','w')
	for z = 1,tmp:size(1) do
		f:write(tmp[z]-1)
		f:write("\n")
	end
	f:close()
	j = sizes[i] + 1
end


