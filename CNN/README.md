==> For training patch based model, execute the following line

th doall.lua -type cuda -batchSize 96 -coefL2 0.0005 -momentum 0.8

==> For training slice based model which trains a CNN to predict whether a given slice has mandible or not, execute the following

th doall.lua -type cuda -batchSize 44 -coefL2 0.0005 -optimization adagrad -mode slice

==> Prediction

==> For predicting the volumes, execute

sh predictVolumes.sh n

==> where n = 1 for testing, n = 2 for validation and n = 3 for training datasets


