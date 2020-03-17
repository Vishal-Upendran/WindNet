#!/bin/bash

niter=300
for i in {1..4}
do
	for j in {1..4}
	do

		for k in {1..5}
		do
		#	for l in {1..4}
		#	do
		#		num1=$(($i+$j))
		#		num2=$(($k+$l))
		#		if (($num1<=$num2))
		#			then
		echo "Iteration started:----------- $i $j"
		python TrainingScript_cv.py $i $j $niter $k
		echo "Iteration done:----------- $i $j"
		#		fi
		#	done
		done
	done
done

