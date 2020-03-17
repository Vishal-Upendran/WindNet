#!/bin/bash
echo Enter channel no
read channelNo
#This script extracts the data and deletes all files except 0000.
bp1="$channelNo"
underscore="_"
AIA="AIA_0"
mkdir -p "$bp1"
for i in {2011..2018}
do
	mkdir -p "$bp1/$i/"
	for j in {01..12}
	do
		tarpath="$i$underscore$AIA$channelNo$underscore$i$j.tar"
		if [ -f "$tarpath" ]; then
			tar -C "$bp1/$i/" -xvf $tarpath
		else 
			continue 
		fi 
		
		for k in {01..31}
		do
			localpath="$bp1/$i/$j/$k/"
			path2="AIA$i$j$k"
			path3="_0000_0$channelNo.npz"
			path="$path2$path3"
			path_6="_0006_0$channelNo.npz"
			path_6_new="$path2$path_6"
			if [ -f "$localpath$path" ]
			then
			#echo "$path"
				ls $localpath | grep -v $path | sed "s|^|$localpath|" | xargs rm
			elif [ -f "$localpath$path_6_new" ]
			then 
				echo "$path_6_new exists" >> nextcad.txt 
				ls $localpath | grep -v $path_6_new | sed "s|^|$localpath|" | xargs rm
			else
				echo "$path does not exist. Do manually" >> notexist.txt
			fi 
		done
	done
done
