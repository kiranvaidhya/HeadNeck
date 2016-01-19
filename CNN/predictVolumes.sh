
if [ $1 == 1 ]
then
	echo "\n---------Predicting Testing Images-------\n"
	th predict.lua -patchSize 13 -batchSize 1000 -path testing
elif [ $1 == 2 ]
then
	echo "\n---------Predicting Validation Images----\n"
	th predict.lua -patchSize 13 -batchSize 1000 -path validation
else
	echo "\n---------Predicting Training Images------\n"
	th predict.lua -patchSize 13 -batchSize 1000 -path training
fi

cd ../HeadNeck/codes
echo "\n---------------Post Processing------------------\n"
python postProcess.py $1
echo "\n---------------Evaluating Dice score------------\n"
python dice.py $1
