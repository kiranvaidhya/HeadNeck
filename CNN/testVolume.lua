require 'cudnn'
require 'cunn'
require 'image'
py = require('fb.python')
require 'xlua'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Testing Images')
cmd:text()
cmd:text('Options')
cmd:option('-batchSize',128)
cmd:option('-patchSize',21)
cmd:text()

params = cmd:parse(arg)
batchSize = params.batchSize
patchSize = params.patchSize

-- LOADING MODEL --

model = torch.load('D45_P13Best/model.net')
sliceModel = torch.load('results/model.net')
---------------------------------------
-- LOADING IMAGE -- 

py.exec([=[
import os
images = []
folders = []
path = '../HeadNeck/data/Testing'
for subdir, dirs, files in os.walk(path):
	for file1 in files:
		if 'normalizedImg' in file1:
			images.append(subdir + '/' + file1)
			folders.append(subdir)

imageNumber = 0
	]=])

-----------------------------------------------------------------
-----------------------------------------------------------------

for imNumber = 1,py.eval('len(images)') do

	py.exec([=[
import numpy as np
import nrrd
import nibabel as nib
from sklearn.feature_extraction import image
print 'Iteration: ', imageNumber+1
print 'Loading Image: ', folders[imageNumber]
img, options = nrrd.read(images[imageNumber])
sizes = img.shape
predictedImage = np.zeros(sizes)
print 'Predicting...'
j = 0
	]=])

	sizes = py.eval('sizes')


	for sliceNumber = 1, sizes[3] do

		py.exec([=[
from sklearn.feature_extraction import image
patch_size = 13
patches = np.zeros((1,patch_size,patch_size))
image_slice = img[:,:,j]
patch = image.extract_patches(image_slice, patch_size, extraction_step = 1)
patch = patch.reshape(patch.shape[0]*patch.shape[1],patch_size,patch_size)
patches = np.append(patches,patch,axis=0)
patches = patches[1:patches.shape[0]]
image_slice = image_slice[156:356,156:356]
j = j+1
	]=])

		local patches = py.eval('patches')
		patches = patches:reshape(patches:size(1),1,patchSize,patchSize)

		-- local slice = py.eval('image_slice')
		-- slice = slice:reshape(1,200,200)
		-- prob, mandibleSwitch = sliceModel:forward(slice:cuda()):max(1)

		-- print(mandibleSwitch[1])

		local predictions = torch.Tensor(patches:size(1)):float()

		-- if mandibleSwitch[1] == 2 then
			for i = 1, patches:size(1),batchSize do
				local batch = patches[{{i,math.min(i+batchSize-1,patches:size(1))}}]:cuda()
				posteriorProbabilities, predictedClasses = model:forward(batch):max(2)
				predictions[{{i,math.min(i+batchSize-1,patches:size(1))}}] = predictedClasses:float()
			end

		-- else
		-- 	predictions = torch.zeros((512-patchSize+1)*(512-patchSize+1)):float()
		-- end

		xlua.progress(sliceNumber,sizes[3])

		py.exec([=[
predictedSlice = pred
predictedSlice = predictedSlice.reshape(512-patch_size+1,512-patch_size+1)
predictedSlice = np.lib.pad(predictedSlice,((patch_size-1)/2,(patch_size-1)/2),'constant',constant_values=0)
predictedImage[:,:,j-1] = predictedSlice
if j == sizes[2]:
	predictedImage[np.where(predictedImage==1)] = 0
	predictedImage[np.where(predictedImage==2)] = 1
	print 'Saving Image: ',folders[imageNumber]
	nrrd.write(folders[imageNumber]+'/prediction.nrrd',predictedImage,options)
	]=],{pred = predictions})
	end
py.exec([=[
imageNumber = imageNumber + 1
]=])
end












