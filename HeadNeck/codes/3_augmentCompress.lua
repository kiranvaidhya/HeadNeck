cmd = torch.CmdLine()
cmd:text()
cmd:text()
cmd:text('Compressing patches')
cmd:text()
cmd:text('Options')
cmd:option('-mode','training','Mode: training | validation')
cmd:option('-patchSize', 11, 'patchSize')
cmd:text()

params = cmd:parse(arg)
mode = params.mode
patchSize = params.patchSize

if mode == 'training' then
	mandibleData = torch.load('tmp/mandibleTrainData.t7')
	mandibleLabels = torch.load('tmp/mandibleTrainLabel.t7')
	zeroData = torch.load('tmp/zeroTrainData.t7')
	zeroLabels = torch.load('tmp/zeroTrainLabel.t7')
elseif mode == 'validation' then
	mandibleData = torch.load('tmp/mandibleValidData.t7')
	mandibleLabels = torch.load('tmp/mandibleValidLabel.t7')
	zeroData = torch.load('tmp/zeroValidData.t7')
	zeroLabels = torch.load('tmp/zeroValidLabel.t7')
end

zeroSize = zeroLabels:size(1)
shuffle = torch.randperm(zeroSize)

zeros = torch.Tensor(zeroData:size())
zerolabels = torch.Tensor(zeroLabels:size())

print '==> Shuffling'

for i = 1,zeroSize do
	zeros[i] = zeroData[shuffle[i]]
	zerolabels[i] = zeroLabels[shuffle[i]]
end

if mode == 'validation' then
	data = torch.cat(mandibleData,zeros,1)
	labels = torch.cat(mandibleLabels,zerolabels,1)
elseif mode == 'training' then
	print '==> Balancing Data'
	data = torch.cat(mandibleData,zeros[{{1,mandibleData:size(1)}}],1)
	labels = torch.cat(mandibleLabels,zerolabels[{{1,mandibleLabels:size(1)}}],1)
end

trsize = labels:size(1)

print '==> Loaded'

shuffle = torch.randperm(trsize)

print '==> Shuffling..'

t = torch.Tensor(data:size())
l = torch.Tensor(labels:size())

for i = 1, trsize do
	t[i] = data[shuffle[i]]
	l[i] = labels[shuffle[i]]
end

if mode == 'validation' then
	t = t[{{torch.floor(trsize/2)-50000,torch.floor(trsize/2)+50000}}]
	l = l[{{torch.floor(trsize/2)-50000,torch.floor(trsize/2)+50000}}]
end
if mode == 'training' then
	-- t = t[{{1,math.min(1000000,t:size(1))}}]
	-- l = l[{{1,math.min(1000000,t:size(1))}}]
end

t = t:float()
l = l:float()

t = t:reshape(t:size(1),1,patchSize,patchSize)

l[torch.eq(l,1)] = 2
l[torch.eq(l,0)] = 1

print('==> Size: ', t:size(1))
print '==> Saving..'

if mode == 'validation' then
	torch.save('s_validData.t7', t)
	torch.save('s_validLabels.t7',l)
elseif mode == 'training' then
	torch.save('s_trainData.t7', t)
	torch.save('s_trainLabels.t7',l)
end


