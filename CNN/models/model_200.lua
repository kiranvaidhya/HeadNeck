----------------------------------------------------------------------
-- This script demonstrates how to define a couple of different
-- models:
--   + linear
--   + 2-layer neural network (MLP)
--   + convolutional network (ConvNet)
--
-- It's a good idea to run this script with the interactive mode:
-- $ th -i 2_model.lua
-- this will give you a Torch interpreter at the end, that you
-- can use to play with the model.
--
-- Clement Farabet
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers
require 'cudnn'

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

-- 8-class problem
noutputs = 2

-- input dimensions
nfeats = 1
width = 200
height = 200
ninputs = nfeats*width*height

----------------------------------------------------------------------
print '==> construct model'

if opt.model == 'convnet' then

   if opt.type == 'cuda' then
      -- a typical modern convolution network (conv+relu+pool)

      model = nn.Sequential()
      -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
      model:add(cudnn.SpatialConvolution(1,64,5,5))
      model:add(cudnn.ReLU())
      model:add(cudnn.SpatialMaxPooling(4,4,4,4))
      model:add(cudnn.SpatialConvolution(64,32,3,3))
      model:add(cudnn.ReLU())
      model:add(cudnn.SpatialMaxPooling(4,4,4,4))
      model:add(cudnn.SpatialConvolution(32,32,3,3))
      model:add(cudnn.ReLU())
      model:add(cudnn.SpatialMaxPooling(2,2,2,2))
      model:add(nn.View(512))
      model:add(nn.Dropout(0.5))
      model:add(nn.Linear(512,200))
      model:add(cudnn.ReLU())
      model:add(nn.Linear(200, noutputs))

      model = model:cuda()
   end
else

   error('unknown -model')

end

----------------------------------------------------------------------
print '==> here is the model:'
print(model)

----------------------------------------------------------------------
-- Visualization is quite easy, using itorch.image().

if opt.visualize then
   if opt.model == 'convnet' then
      if itorch then
    print '==> visualizing ConvNet filters'
    print('Layer 1 filters:')
    --itorch.image(model:get(1).weight)
         image.display(model:get(1).weight)
    print('Layer 2 filters:')
    --itorch.image(model:get(5).weight)
         image.display(model:get(5).weight)
      else
    print '==> To visualize filters, start the script in itorch notebook'
      end
   end
end
