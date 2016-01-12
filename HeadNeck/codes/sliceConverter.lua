py = require('fb.python')
require 'image'

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Converting slices to make them compatible with Lua')
cmd:text()
cmd:text('Options')
cmd:option('-mode','training','Mode: training | validation')
cmd:text()

-- parse input params
params = cmd:parse(arg)

if params.mode == 'validation' then
py.exec([=[
import numpy as np
slices = np.load('slices/validation/slices.npy')
truths = np.load('slices/validation/truths.npy')
	]=])
else
py.exec([=[
import numpy as np
slices = np.load('slices/training/slices.npy')
truths = np.load('slices/training/truths.npy')
	]=])
end

slices = py.eval('slices')
truths = py.eval('truths')

if params.mode == 'training' then

	rr = torch.range(1,slices:size(1)):long()
	idx = torch.eq(truths,2)
	mandibles = slices:index(1,rr[idx])

	idx1 = torch.eq(truths,1)
	zeros = slices:index(1,rr[idx1])

	flipMandibles = image.hflip(mandibles)
	rot5Mandibles = image.rotate(mandibles,10*math.pi/180,'bilinear')
	rotm5Mandibles = image.rotate(mandibles,-10*math.pi/180,'bilinear')

	print '==> Flipping and Rotating..'

	mandibles = mandibles:cat(flipMandibles,1)
	mandibles = mandibles:cat(rot5Mandibles,1)
	mandibles = mandibles:cat(rotm5Mandibles,1)

	slices = torch.cat(zeros,mandibles,1)

	zeros = torch.ones(zeros:size(1))
	ones = torch.ones(mandibles:size(1))*2
	truths = torch.cat(zeros,ones,1)

	print '==> Scaling and Cropping..'

	tmp = image.scale(slices,300,300)
	tmp = image.crop(tmp,"c",200,200)

	slices = torch.cat(slices,tmp,1)
	truths = torch.cat(truths,truths,1)
end

slices = slices:reshape(slices:size(1),1,200,200)

shuffle = torch.randperm(truths:size(1))

tslices = torch.zeros(slices:size())
ttruths = torch.zeros(truths:size())

for i = 1,slices:size(1) do
	tslices[i] = slices[shuffle[i]]
	ttruths[i] = truths[shuffle[i]]
end

if params.mode == 'validation' then
	torch.save('slices/slices_validation.t7',tslices)
	torch.save('slices/truths_validation.t7',ttruths)
else
	torch.save('slices/slices_training.t7',tslices)
	torch.save('slices/truths_training.t7',ttruths)
end