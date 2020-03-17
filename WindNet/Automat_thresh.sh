#!/bin/bash
delay=2
history=4
n_iter2=300
for cv in {1..5}
do
	for tv1 in 100 500 1000
	do
		for tv2 in 10000 20000 40000
		do
			python TrainingScript_cv.py $delay $history $n_iter2 $cv $tv1 $tv2
		done
	done
done
