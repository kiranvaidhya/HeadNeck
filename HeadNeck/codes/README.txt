preProcess.py normalizes the images using Z-score.
combineTruth.py combines the truths into one single ground truth file
1_extractPatches.py extracts patches from the normalized images
2_luaPatchConverter.lua converts numpy file into t7 file
3_compressPatchesLua.lua converts the t7 file into float and compresses the entire data.

The sequence is from 1 -> 2 -> 3
Be careful to specify the -mode argument for 3_compressPatchesLua.lua

Training patches are stored in 'patches'
Validation patches are stored in 'validationPatches'
