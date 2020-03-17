import numpy as np
import os
import csv
from collections import OrderedDict as DC
import sys
data_path = "../OMNI/data_OMNI.lst"
formatter_path = '../OMNI/dataformat_OMNI.fmt'

"""
	We are using NASA OMNIWEB (OMNI2) Dataset available at : https://omniweb.gsfc.nasa.gov/form/dx1.html
	We have two files- one which contains the data, and other which contains the format specifier.
	This file will combine the two into a dictionary for easy access.
"""

dataset = np.loadtxt(data_path)

print "Data loaded. Data having shape " + str(dataset.shape)

with open(formatter_path) as f:
	reader = csv.reader(f)
	formlist = list(reader)
print formlist[-1]
"""
	The OMNI dataset contains headers, etc. It looks like:
		['  FORMAT OF THE SUBSETTED FILE']
		['    ']
		['    ITEMS                      FORMAT   ']
		['     ']
		[' 1 YEAR                          I4        ']
		[' 2 DOY                           I4        ']
		[' 3 Hour                          I3        ']
		[' 4 BX', ' nT (GSE', ' GSM)             F6.1      ']
		[' 5 BY', ' nT (GSE)                  F6.1      ']
		[' 6 BZ', ' nT (GSE)                  F6.1      ']
		[' 7 BY', ' nT (GSM)                  F6.1      ']
		[' 8 BZ', ' nT (GSM)                  F6.1      ']
		[' 9 RMS_BX_GSE', ' nT                F6.1      ']
		['10 RMS_BY_GSE', ' nT                F6.1      ']
		['11 RMS_BZ_GSE', ' nT                F6.1      ']
		['12 SW Plasma Temperature', ' K      F9.0      ']
		['13 SW Proton Density', ' N/cm^3     F6.1      ']
		['14 SW Plasma Speed', ' km/s         F6.0      ']
		['15 sigma-T', 'K                     F9.0      ']
		['16 sigma-n', ' N/cm^3)              F6.1      ']
		['17 sigma-V', ' km/s                 F6.0      ']
		[]
		['This file was produced by SPDF OMNIWeb Plus service']
	We will need to remove the header and the footer. Hence, we will have to iterate thrdough 4:len(formlist)--2.
	Also, we need the strings alone, and not the numbers, units, etc for the data. Hence, we will have to strip the data.
"""
keylist=[]
Data= DC()
iter_val = 0
formlist = formlist[4:] #was 4:len(formlist)-2
for key in formlist:
	key = key[0].strip(str(np.arange(1,18))).split("    ",1)[0]
	if iter_val not in [6,7]:
		keylist.append(key)
		Data[key] = dataset[:,iter_val]
		print key
	iter_val = iter_val + 1
	"""
		To remove the numerals, we strip the text. And then to remove the stuff after BX.., we use split.
		Please note I have also downloaded By,Bz(GSE) and GSM, and since I have stddev for only GSE, I will discard the GSM values
		(they are anyway related by a transformation). So, index numbers 6 and 7 are gone.
	"""
print "Data stripping..... done"
print "Dictionary formation....... done"

np.save('../OMNI/OMNIWEB_data_dict',Data)
