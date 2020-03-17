#!/bin/bash
#Purpose of the script: Clean the AIA data, clean the OMNI data, combine and partition them.
channelNo=211
echo "Process.py..... initiated."
python Process.py $channelNo
#echo "DataCombiner.py..... initiated."
#python DataCombiner.py
echo "DataCleaner.py..... initiated."
python DataCleaner.py $channelNo
echo "DataCreator_cv.py..... initiated."
python DataCreator_cv.py $channelNo
#echo "Generate_cv_from_folds.py.....initiated"
#python Generate_cv_from_folds.py $channelNo
