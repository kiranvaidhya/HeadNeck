py = require('fb.python')
require 'cudnn'
require 'optim'
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


model = torch.load('D45_P13Best/model.net')
sliceModel = torch.load('results/model.net')

model:evaluate()
sliceModel:evaluate()

py.exec([=[

from sklearn.feature_extraction import image as extractor
import nrrd
import os
import getopt
import sys
import numpy as np
from scipy.ndimage import zoom

patchSize = 13

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

sizes = []
nslices = 0

i = 0
]=])

for imageIterator = 1,py.eval('len(images)') do

	py.exec([=[

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
	i = i+1
	img, options = nrrd.read(images[i])
folder = folders[i]

slices = np.zeros((1,200,200))
ground_truths = np.zeros((1))

for j in xrange(img.shape[2]):
	imgSlice = img[156:356,156:356,j]
	truthSlice = truth[:,:,j]
	imgSlice = imgSlice.reshape(1,200,200)
	slices = np.append(slices,imgSlice,axis=0)
	ground_truths = np.append(ground_truths,np.unique(truthSlice).shape[0])

slices = slices[1:slices.shape[0]]
ground_truths = ground_truths[1:ground_truths.shape[0]]

nslices = nslices + img.shape[2]
sizes.append(nslices)

predictedImage = np.zeros(img.shape)

print 'Mandible: ', np.sum((ground_truths==2).astype(int))

i = i + 1
sliceIterator = 0

	]=])

	slices = py.eval('slices')
	slices = slices:reshape(slices:size(1),1,200,200)
	truths = py.eval('ground_truths')


	y = torch.zeros(slices:size(1)):cuda()

	outputs = torch.zeros(slices:size(1),2):cuda()
	for i = 1,slices:size(1) do
		tmp1, tmp2 = sliceModel:forward(slices[i]:cuda()):max(1)
		outputs[i] = sliceModel:forward(slices[i]:cuda())
		y[i] = tmp2
	end

	classes = {'1','2'}

	confusion = optim.ConfusionMatrix(classes)

	for i = 1,slices:size(1) do
		confusion:add(outputs[i],truths[i])
	end

	print(confusion)

	-- f = io.open('prem'..tostring(imageIterator)..'txt','w')
	-- for z = 1,y:size(1) do
	-- 	f:write(y[z]-1)
	-- 	f:write("\n")
	-- end
	-- f:close()


	for sliceIterator = 1,slices:size(1) do

		if y[sliceIterator] == 1 then

			predictions = torch.zeros((512-patchSize+1)*(512-patchSize+1))
			py.exec([=[
sliceIterator = sliceIterator + 1
				]=])

		else
			py.exec([=[
patches = np.zeros((1,patchSize,patchSize))
patch = extractor.extract_patches(img[:,:,sliceIterator], patchSize, extraction_step = 1)
patch = patch.reshape(patch.shape[0]*patch.shape[1],patchSize,patchSize)
patches = np.append(patches,patch,axis=0)
patches = patches[1:patches.shape[0]]
sliceIterator = sliceIterator + 1
			]=])

			local patches = py.eval('patches')
			patches = patches:reshape(patches:size(1),1,patchSize,patchSize)

			predictions = torch.Tensor(patches:size(1)):float()

			for i = 1, patches:size(1),batchSize do
				local batch = patches[{{i,math.min(i+batchSize-1,patches:size(1))}}]:cuda()
				posteriorProbabilities, predictedClasses = model:forward(batch):max(2)
				predictions[{{i,math.min(i+batchSize-1,patches:size(1))}}] = predictedClasses:float()
			end
		end

		py.exec([=[
predictedSlice = pred
predictedSlice = predictedSlice.reshape(512-patchSize+1,512-patchSize+1)
predictedSlice = np.lib.pad(predictedSlice,((patchSize-1)/2,(patchSize-1)/2),'constant',constant_values=0)
predictedImage[:,:,sliceIterator-1] = predictedSlice
if sliceIterator == img.shape[2]:
	predictedImage[np.where(predictedImage==1)] = 0
	predictedImage[np.where(predictedImage==2)] = 1
	print 'Saving Image: ',folders[i-1]
	nrrd.write(folders[i-1]+'/prediction.nrrd',predictedImage,options)
	]=],{pred = predictions})

	xlua.progress(sliceIterator,py.eval('img.shape[2]'))

	end
	
end



