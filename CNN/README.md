# Training a CNN for Mandible Segmentation

## Patch based mandible/background classifier

`th doall.lua -type cuda -batchSize 96 -coefL2 0.0005 -momentum 0.8`

## Training slice based model which trains a CNN to predict whether a given slice has mandible or not

`th doall.lua -type cuda -batchSize 44 -coefL2 0.0005 -optimization adagrad -mode slice`

## Prediction

`sh predictVolumes.sh n`

where n = 1 for testing, n = 2 for validation and n = 3 for training datasets


