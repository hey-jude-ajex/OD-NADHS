#!/bin/bash
#percentages=('5%' '10%' '15%' '20%' '25%' '30%' '35%' '40%' '45%' '50%')
#
#
#for percentage in ${percentages[*]}
#do
#	echo $percentage
#	cp ../dataset/MNIST/miss_dimensions/miss_dimensions_$percentage.txt ./dataset/miss_dimensions.txt
#	cp ../dataset/MNIST/train/$percentage/train_noise.txt ./dataset/train_noise.txt	
#	cp ../dataset/MNIST/test/$percentage/test_noise.txt ./dataset/test_noise.txt
#	cp ../dataset/MNIST/softmax_W.txt ./dataset/softmax_W.txt
#	cp ../dataset/MNIST/softmax_b.txt ./dataset/softmax_b.txt
#
#	echo $percentage >> result.txt
#	/usr/local/bin/python3.6 main.py >> result.txt
#	echo '/////////////' >> result.txt
#done


for((i=0;i<10;i++))
do 
	echo $i >> result.txt
	E:/Programs/Python/Python37/python main.py >> result.txt
done
