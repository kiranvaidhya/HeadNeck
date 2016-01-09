echo "------------ Extracting Patches ------------"
python mandibleExtraction.py -p $1
python mandibleExtraction.py -p $1 -v
python zeroExtraction.py -p $1
python zeroExtraction.py -p $1 -v
echo "\n------------ MandibleConverter - Training-----------------\n"
th 2_mandibleConverter.lua -patchSize $1 -i2 23
echo "\n------------ MandibleConverter - Validation-----------------\n"
th 2_mandibleConverter.lua -patchSize $1 -validation
echo "\n------------ ZeroConverter - Validation-----------------\n"
th 2_zeroConverter.lua -patchSize $1 -i2 23
echo "\n------------ ZeroConverter - Validation-----------------\n"
th 2_zeroConverter.lua -patchSize $1 -validation
echo "\n------------ Compressing and Augmenting Training Patches -----------------\n"
th 3_augmentCompress.lua -patchSize $1
echo "\n------------ Compressing Validation Patches -----------------\n"
th 3_augmentCompress.lua -patchSize $1 -mode validation
