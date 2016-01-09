require 'cudnn'
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

model = torch.load('results/model.net')
---------------------------------------
-- LOADING IMAGE -- 
py.exec([=[
import numpy as np
import nrrd
import nibabel as nib

from sklearn.feature_extraction import image
img, options = nrrd.read('../HeadNeck/data/Testing/0522c0328/normalizedImg.nrrd')
sizes = img.shape
predictedImage = np.zeros(sizes)

j = 0
k = 0

]=])

sizes = py.eval('sizes')

for sliceNumber = 1, sizes[3] do
	py.exec([=[
from sklearn.feature_extraction import image

patch_size = 13

patches = np.zeros((1,patch_size,patch_size))

print '==> Extracting patches..'

image_slice = img[:,:,j]
patch = image.extract_patches(image_slice, patch_size, extraction_step = 1)
patch = patch.reshape(patch.shape[0]*patch.shape[1],patch_size,patch_size)
patches = np.append(patches,patch,axis=0)
patches = patches[1:patches.shape[0]]
print 'Slice: ', j
j = j+1
]=])
	print '==> Patches extracted from Python'

	local slice = py.eval('patches')
	slice = slice:reshape(slice:size(1),1,patchSize,patchSize)

	local predictions = torch.Tensor(slice:size(1)):float()

	for i = 1, slice:size(1),batchSize do
		local batch = slice[{{i,math.min(i+batchSize-1,slice:size(1))}}]:cuda()
		posteriorProbabilities, predictedClasses = model:forward(batch):max(2)
		-- predictedClasses = batch[{{},{},10,10}]
		predictions[{{i,math.min(i+batchSize-1,slice:size(1))}}] = predictedClasses:float()
		xlua.progress(i,slice:size(1))
	end
	py.exec([=[
predictedSlice = pred
predictedSlice = predictedSlice.reshape(512-patch_size+1,512-patch_size+1)
predictedSlice = np.lib.pad(predictedSlice,((patch_size-1)/2,(patch_size-1)/2),'constant',constant_values=0)
predictedImage[:,:,j-1] = predictedSlice
print 'predictedSlice: ', j-1
if j == sizes[2]:
	predictedImage[np.where(predictedImage==1)] = 0
	predictedImage[np.where(predictedImage==2)] = 1
	affine=[[-1,0,0,0],[0,-1,0,0],[0,0,1,0],[0,0,0,1]]
	finalImg = nib.Nifti1Image(predictedImage, affine)
	finalImg.set_data_dtype(np.int32)
	nib.save(finalImg,'prediction1.nii')
]=],{pred = predictions})
end













