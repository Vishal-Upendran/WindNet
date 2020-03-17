## Data preprocessing pipeline
The data consists of SDO AIA imagery data and NASA OMNIWEB data. The pre-processing has a couple of steps as shown below:

A. Downloaded AIA is data is stored in `AIA/` folder. This must be log scaled, clipped and clamped to obtain the input to our model.
B. The modified AIA and the daily averaged OMNI data must match the time stamps -- thus missing datapoints must be removed. The OMNI data has no missing data in the daily averaged dataset. Thus, only the AIA needs to be taken care of.
C. Dataset partitioning into folds is done, and stored at `Data/channel_no/Fold`.
D. Generate the cross validation set from folds, and save to `Data/channel_no/CV_`
E. This data can now be used to perform prediction.

So the way to process data is:

1. Have the AIA data in `AIA/`
2. Pre-pre process and perform step **A** using `Process.py`.
3. Clean the OMNI data by making it into a dictionary using `DataCombiner.py`.
4. Perform step **B** using `Datacleaner.py` and remove OMNI data which does not have any corresponding AIA data. We obtain OMNI data in multiple files with corresponding years.
5. Perform step **C** using `Datacreator_cv.py`.
6. Once the cross validation folds are created, generate cross validation set using `Generate_cv_from_folds.py`.

To obtain the masks, change the flag `mask` to `True` and perform Step-6 alone.
