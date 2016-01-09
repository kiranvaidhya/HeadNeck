# echo "E1"
# th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.00005 -coefL2 0.00005 -save D22_P13 -size small > D22_P13/details.txt
# echo "E2"
# th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.00005 -coefL2 0.00005 -save D23_P13 -learningRate 1 -size small > D23_P13/details.txt
# echo "E3"
# th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.00005 -coefL2 0.00005 -save D24_P13 -learningRate 0.1 -size small > D24_P13/details.txt
# echo "E4"
# th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.00005 -coefL2 0.00005 -save D25_P13 -learningRate 0.0001 -size small > D25_P13/details.txt
# echo "E5"
# th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.00005 -coefL2 0.00005 -save D26_P13 -learningRate 0.0005 -size small > D26_P13/details.txt
# echo "E6"
# th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.00005 -coefL2 0.00005 -save D27_P13 -learningRate 0.5 -size small > D27_P13/details.txt
# echo "E7"
# th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.00005 -coefL2 0.00005 -save D28_P13 -learningRate 1 -momentum 0.9 -size small > D28_P13/details.txt
# echo "E8"
# th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.00005 -coefL2 0.00005 -save D29_P13 -learningRate 0.1 -momentum 0.9 -size small > D29_P13/details.txt
# echo "E9"
# th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.00005 -coefL2 0.00005 -save D30_P13 -learningRate 0.0001 -momentum 0.9 -size small > D30_P13/details.txt
# echo "E10"
# th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.00005 -coefL2 0.00005 -save D31_P13 -learningRate 0.0005 -momentum 0.9 -size small > D31_P13/details.txt
# echo "E11"
# th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.00005 -coefL2 0.00005 -save D22_P13 -momentum 0.9 -size small > D32_P13/details.txt


echo "E1"

th doall.lua -type cuda -batchSize 96 -loss margin -coefL2 5 -save D33_P13 -learningRate 0.0005 -size small > D33_P13/details.txt
echo "E2"
th doall.lua -type cuda -batchSize 96 -loss margin -coefL2 0.5 -save D34_P13 -learningRate 0.0005 -size small > D34_P13/details.txt
echo "E3"
th doall.lua -type cuda -batchSize 96 -loss margin -coefL2 0.05 -save D35_P13 -learningRate 0.0005 -size small > D35_P13/details.txt
echo "E4"
th doall.lua -type cuda -batchSize 96 -loss margin -coefL2 0.005 -save D36_P13 -learningRate 0.0005 -size small > D36_P13/details.txt
echo "E5"
th doall.lua -type cuda -batchSize 96 -loss margin -coefL2 0.0005 -save D37_P13 -learningRate 0.0005 -size small > D37_P13/details.txt
echo "E6"
th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.00005 -coefL2 0.00005 -save D38_P13 -learningRate 0.0005 -size small > D38_P13/details.txt
echo "E7"
th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 5 -coefL2 5 -save D39_P13 -learningRate 0.0005 -size small > D39_P13/details.txt
echo "E8"
th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.5 -coefL2 0.5 -save D40_P13 -learningRate 0.0005 -size small > D40_P13/details.txt
echo "E9"
th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.05 -coefL2 0.05 -save D41_P13 -learningRate 0.0005 -size small > D41_P13/details.txt
echo "E10"
th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.005 -coefL2 0.005 -save D42_P13 -learningRate 0.0005 -size small > D42_P13/details.txt
echo "E11"
th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.0005 -coefL2 0.0005 -save D43_P13 -learningRate 0.0005 -size small > D43_P13/details.txt
echo "E12"
th doall.lua -type cuda -batchSize 96 -loss margin -coefL1 0.00005 -coefL2 0.00005 -save D44_P13 -learningRate 0.0005 -size small > D44_P13/details.txt


