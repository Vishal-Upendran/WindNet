#!/bin/bash

for i in {1..4}
do
	for j in {1..4}
	do	

		for k in {1..5}
		do 
		echo "Iteration started:----------- $i $j $k" 
		python SWPrediction_XGboost_cv.py $i $j $k 
		python SWPrediction_SWSVM_cv.py $i $j $k
		python SWPrediction_NaiveMean_cv.py $i $j $k
		python SWPrediction_Persistence_cv.py $i $j $k
		echo "Iteration done:----------- $i $j $k" 
		done
	done
done
