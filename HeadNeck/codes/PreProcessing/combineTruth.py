import os
import nrrd
import numpy as np

path = '../../data/'

images = []
brainStems = []
chiasms = []
mandibles = []
leftOpticNerves = []
rightOpticNerves = []
leftParotids = []
rightParotids = []
leftSubmandibulars = []
rightSubmandibulars = []

imageFolders = []
truthFolders = []

for subdir, dirs, files in os.walk(path):
	for file1 in files:
		if 'img' in file1:
			images.append(subdir + '/' + file1)
			imageFolders.append(subdir)
		elif 'BrainStem' in file1:
			brainStems.append(subdir + '/' + file1)
			truthFolders.append(subdir)
		elif 'Chiasm' in file1:
			chiasms.append(subdir + '/' + file1)
		elif 'Mandible' in file1:
			mandibles.append(subdir + '/' + file1)
		elif 'OpticNerve_L' in file1:
			leftOpticNerves.append(subdir + '/' + file1)
		elif 'OpticNerve_R' in file1:
			rightOpticNerves.append(subdir + '/' + file1)
		elif 'Parotid_L' in file1:
			leftParotids.append(subdir + '/' + file1)
		elif 'Parotid_R' in file1:
			rightParotids.append(subdir + '/' + file1)
		elif 'Submandibular_L' in file1:
			leftSubmandibulars.append(subdir + '/' + file1)
		elif 'Submandibular_R' in file1:
			rightSubmandibulars.append(subdir + '/' + file1)


for i in xrange(len(images)):

	print '==> Iteration: ', i+1
	image, options = nrrd.read(images[i])
	brainStem, options = nrrd.read(brainStems[i])
	chiasm, options = nrrd.read(chiasms[i])
	try:
		mandible, options = nrrd.read(mandibles[i])
	except:
		pass
	leftOpticNerve, options = nrrd.read(leftOpticNerves[i])
	rightOpticNerve, options = nrrd.read(rightOpticNerves[i])
	leftParotid, options = nrrd.read(leftParotids[i])
	rightParotid, options = nrrd.read(rightParotids[i])
	try:
		leftSubmandibular, options = nrrd.read(leftSubmandibulars[i])
		rightSubmandibular, options = nrrd.read(rightSubmandibulars[i])
	except:
		pass

	combinedTruth = brainStem
	combinedTruth[np.where(chiasm == 1)] = 2
	try:
		combinedTruth[np.where(mandible == 1)] = 3
	except:
		pass
	combinedTruth[np.where(leftOpticNerve == 1)] = 4
	combinedTruth[np.where(rightOpticNerve == 1)] = 5
	combinedTruth[np.where(leftParotid == 1)] = 6
	combinedTruth[np.where(rightParotid == 1)] = 7
	try:
		combinedTruth[np.where(leftSubmandibular == 1)] = 8
		combinedTruth[np.where(rightSubmandibular == 1)] = 9
	except:
		pass

	nrrd.write(truthFolders[i] + '/' + 'combinedTruth.nrrd', combinedTruth)




