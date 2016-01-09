require 'nn'
require 'image'
require 'xlua'

if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'full', 'how many samples do we load: small | full | extra')
   cmd:text()
   opt = cmd:parse(arg or {})
end

trainData = torch.load('../HeadNeck/codes/s_trainData.t7')
trainLabels = torch.load('../HeadNeck/codes/s_trainLabels.t7')
testData = torch.load('../HeadNeck/codes/s_validData.t7')
testLabels = torch.load('../HeadNeck/codes/s_validLabels.t7')

trsize = trainData:size(1)
tesize = testData:size(1)

trainData = {
   data = trainData,
   labels = trainLabels,
   size = function() return trsize end
}

testData = {
   data = testData,
   labels = testLabels,
   size = function() return tesize end
}

if opt.size == 'small' then
	print '==> Using a subset(20%) of the entire dataset for faster experiments'

	-- print('==> TestData - ', math.floor(testData.data:size(1)/5))
	-- testData.data = trainData.data[{{100000,110000}}]
	-- testData.labels = trainData.labels[{{100000,110000}}]
	-- testData.size = function() return testData.data:size(1) end

	
	trainData.data = trainData.data[{{1,math.floor(trainData.data:size(1)/5)}}]
	trainData.labels = trainData.labels[{{1,math.floor(trainData.labels:size(1)/5)}}]
	trainData.size = function() return trainData.data:size(1) end

	print('==> TrainData - ', trainData:size())
	
	print('==> TestData - ', math.floor(testData.data:size(1)/5))
	testData.data = testData.data[{{1,math.floor(testData.data:size(1)/5)}}]
	testData.labels = testData.labels[{{1,math.floor(testData.labels:size(1)/5)}}]
	testData.size = function() return testData.data:size(1) end
end

-- print '==> Normalizing Locally'

-- neighborhood = image.gaussian1D(13)

-- normalization = nn.SpatialContrastiveNormalization(1, neighborhood, 1):float()

-- channels = {'g'}

-- Normalize all channels locally:
-- for c in ipairs(channels) do
--    for i = 1,trainData.data:size(1) do
--       trainData.data[{ i,{c},{},{} }] = normalization:forward(trainData.data[{ i,{c},{},{} }])
--       -- xlua.progress(i,trainData.data:size(1))
--    end
--    for i = 1,testData.data:size(1) do
--       testData.data[{ i,{c},{},{} }] = normalization:forward(testData.data[{ i,{c},{},{} }])
--       -- xlua.progress(i,testData.data:size(1))
--    end
-- end