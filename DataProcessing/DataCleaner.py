import numpy as np
import os
from glob import glob
from itertools import izip,chain
import datetime as dt
import sys
#Enter the channel no.
channelNo= int(sys.argv[1])
sdo_main_path = '../Processed_data/'+str(channelNo)+'/_20'
sw_main_path = '../OMNI/OMNIWEB_data_dict.npy'
"""
	Our SDO files are of the format: ---- sdo_multichannel_201005130000.bin ---, or better written as:
	sdo_multichannel_yearmonthday0000.bin. Hence, we will strip this value, obtain the 'day of the year', and
	make a processed data. We are doing this since SDO data has some days missing.
"""
swdata = np.load(sw_main_path,allow_pickle=True).tolist()
"""
	Algorithm:
		1. Strip out the date as dd-mm-yy from the sdo filenames.
		2. Please note SW data is labelled continuously- an array ranging from start point of SDO to end point of SDO.
		3. SW is given as year-day number of the year-... Hence we must convert the yyyy-mm-dd to this and compare.
		4. If date_year(sdo)==date_year(sw):
				if date_day(sdo)==date_day(sw):
					retain sw
				else if date_day(sdo)>date_day(sw):
					iterate till same day is reached for sw.
					retain the new value.
				else
					do nothing, as we are processing sw data. No discontinuties in SW (AS OF NOW!)
			else if date_year(sdo)>date_year(sw):
					iterate till same year is reached for sw.
					do comparisons.
					if date_day(sdo)==date_day(sw):
						retain sw
					else if date_day(sdo)>date_day(sw):
						iterate till same day is reached for sw.
						retain the new value.
					else
						do nothing, as we are processing sw data. No discontinuties in SW (AS OF NOW!)
			else:
				Do nothing. This issue shouldn't crop up.
"""
for k in swdata.keys():
	print k
itervar=0
j=0
for yr  in [11,12,13,14,15,16,17,18]:
	base_path = sdo_main_path+str(yr)
	filenames  = sorted(glob(base_path+'*.npy'))
	#print filenames
	DataPart=[]
	for name in filenames:
		name = name.split('_')[-1]
		name = name.split('.')[0] #We almost have our yyyymmdd format
		yy = int(name[:4])
		mm = int(name[4])*10+int(name[5])
		dd = int(name[6])*10+int(name[7])
		sdo_date = dt.datetime(yy,mm,dd)-dt.datetime(yy,1,1)
		sdo_date = sdo_date.days + 1
		#print str(name),
		#print "..."+str(int(swdata['DOY'][itervar])),
		#print "---"+str(yy),
		#print "..."+str(int(swdata['YEAR'][itervar]))
		if int(swdata['YEAR'][itervar])==yy:
			if int(swdata['DOY'][itervar])==sdo_date:
				DP=[]
				print str(sdo_date),
				print "..."+str(int(swdata['DOY'][itervar]))
				for k in swdata.keys():
					DP.append(swdata[k][itervar])
				DP=np.asarray(DP)
				DataPart.append(DP)
				itervar = itervar + 1
			elif int(swdata['DOY'][itervar])<sdo_date:
				while (int(swdata['DOY'][itervar])!=sdo_date):
					itervar = itervar + 1
				DP=[]
				for k in swdata.keys():
					DP.append(swdata[k][itervar])
				DP=np.asarray(DP)
				DataPart.append(DP)
				itervar = itervar + 1
			else:
				_ = 'lol'
		elif int(swdata['YEAR'][itervar])<yy:
			while (int(swdata['YEAR'][itervar])!=yy):
					itervar = itervar + 1

			if int(swdata['DOY'][itervar])==sdo_date:
				DP=[]
				print str(sdo_date),
				print "..."+str(int(swdata['DOY'][itervar]))
				for k in swdata.keys():
					DP.append(swdata[k][itervar])
				DP=np.asarray(DP)
				DataPart.append(DP)
				itervar = itervar + 1
			elif int(swdata['DOY'][itervar])<sdo_date:
				while (int(swdata['DOY'][itervar])!=sdo_date):
					itervar = itervar + 1
				DP=[]
				for k in swdata.keys():
					DP.append(swdata[k][itervar])
				DP=np.asarray(DP)
				DataPart.append(DP)
				itervar = itervar + 1
			else:
				pass


	DataPart = np.asarray(DataPart)
	print DataPart.shape
	print len(filenames)
	np.savetxt(str(channelNo)+'OMNI_20'+str(yr)+'.txt',DataPart)
