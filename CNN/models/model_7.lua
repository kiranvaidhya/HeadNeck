require 'torch'   -- torch
require 'image'   -- for image transforms
-- require 'nn'      -- provides all sorts of trainable modules/layers
require 'cudnn'
-- require 'requ'

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('SVHN Model Definition')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-model', 'convnet', 'type of model to construct: linear | mlp | convnet')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:text()
   opt = cmd:parse(arg or {})
end

----------------------------------------------------------------------
print '==> define parameters'

-- 4-class problem
noutputs = 2

-- input dimensions
nfeats = 1
width = 7
height = 7
ninputs = nfeats*width*height

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,256}
filtsize = 3
poolsize = 2
normkernel = image.gaussian1D(7)

----------------------------------------------------------------------
print '==> construct model'

if opt.model == 'convnet' then

   if opt.type == 'cuda' then
      -- a typical modern convolution network (conv+relu+pool)

      model = nn.Sequential()
      model:add(cudnn.SpatialConvolution(nfeats, nstates[1], filtsize, filtsize))
      model:add(cudnn.ReLU())
      -- model:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
      -- model:add(nn.ReLU())
      model:add(cudnn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))
      model:add(nn.View(nstates[2]))
      model:add(nn.Dropout(0.5))
      -- model:add(nn.Linear(nstates[3],nstates[4]))
      -- model:add(cudnn.ReLU())
      model:add(nn.Linear(nstates[2],noutputs)) 

      model = model:cuda()

      model = require('weight-init')(model,'xavier')

   end
else

   error('unknown -model')

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

