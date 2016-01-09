py = require('fb.python')

cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Converting patches to make them compatible with Lua')
cmd:text()
cmd:text('Options')
cmd:option('-patchSize',11,'Patch Size')
cmd:option('-validation',false,'Validation option')
cmd:option('-i1', 1, 'Initial image number')
cmd:option('-i2', 5, 'Final image number')
cmd:text()

-- parse input params
params = cmd:parse(arg)
patchSize = params.patchSize
validation = params.validation
i1 = params.i1
i2 = params.i2

if validation == false then
	py.exec([=[

patch_root = 'patches/'
import numpy as np
patches = []
truths = []
for i in xrange(23):
	patches.append(np.load(patch_root + 'patches_' + str(i) + '.npy'))
	truths.append(np.load(patch_root + 'truths_' + str(i) + '.npy'))

	]=])
else
	py.exec([=[

patch_root = 'validationPatches/'
import numpy as np
patches = []
truths = []
for i in xrange(5):
	patches.append(np.load(patch_root + 'patches_' + str(i) + '.npy'))
	truths.append(np.load(patch_root + 'truths_' + str(i) + '.npy'))
	]=])
end

torch_patches = torch.Tensor(1,patchSize,patchSize)
torch_truths = torch.Tensor(1)

for i = i1,i2 do
	print('==> Appending Image: ', i)
	torch_patches = torch.cat(torch_patches, py.eval('patches[i-1]'),1)
	torch_truths = torch.cat(torch_truths, py.eval('truths[i-1]'),1)
end

torch_patches = torch_patches[{{2,torch_patches:size(1)}}]
torch_truths = torch_truths[{{2,torch_truths:size(1)}}]

if validation == true then
	torch.save('tmp/mandibleValidData.t7', torch_patches)
	torch.save('tmp/mandibleValidLabel.t7', torch_truths)
else
	torch.save('tmp/mandibleTrainData.t7', torch_patches)
	torch.save('tmp/mandibleTrainLabel.t7', torch_truths)
end
