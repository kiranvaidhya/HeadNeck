require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
-- require 'cudnn'
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
noutputs = 4

-- input dimensions
nfeats = 1
width = 21
height = 21
ninputs = nfeats*width*height

-- hidden units, filter sizes (for ConvNet only):
nstates = {64,64,576,100}
filtsize = 5
poolsize = 2
normkernel = image.gaussian1D(7)

----------------------------------------------------------------------
print '==> construct model'

if opt.model == 'convnet' then

   if opt.type == 'cuda' then
      -- a typical modern convolution network (conv+relu+pool)

      model = nn.Sequential()
      model:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(2,2,2,2))	
      model:add(nn.SpatialConvolutionMM(nstates[1],nstates[2],3,3))
      model:add(nn.ReLU())
      model:add(nn.SpatialMaxPooling(2,2,2,2))
      model:add(nn.View(nstates[3]))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(nstates[3],nstates[4]))
      model:add(nn.ReLU())
      model:add(nn.Linear(nstates[4],noutputs))
      model:add(nn.ReLU())

   end
else

   error('unknown -model')

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

