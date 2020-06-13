### Solar wind prediction using deep learning

This repository contains codes for the work [`Solar wind prediction using deep learning`](https://agupubs.onlinelibrary.wiley.com/doi/abs/10.1029/2020SW002478). If you are using this code (in part or in entirety), or the results and conclusions from this study, do cite us as:

<font color = '#559678'>  Upendran, V., Cheung, M. C. M., Hanasoge, S., & Krishnamurthi, G. ( 2020). Solar wind prediction using deep learning. Space Weather, 18, e2020SW002478. https://doi.org/10.1029/2020SW002478 </font>

The repository is divided into multiple parts as follows:

1. Data download section.
2. Data processing section.
3. Benchmark modelling section.
4. WindNet modelling section.
5. Visualization and plotting.

Each section has its own README file. Please make sure the required steps are all followed in order to replicate the results correctly. For using this model for a future work, make sure the data format is correctly done.

-------------------------------------

#### Data download
We use the SDO ML dataset from Galvez et al (2019). This must be downloaded, and one obtains .tar files. These must be extracted in the format `AIA/channel_no/yy/mm/dd/`. Since we use only 1 datapoint per day, the remanining ones must be deleted. This extraction procedure is performed by `BulkDownload.sh` script, once the tar files are available. The OMNI data must be present in `OMNI/` folder.

There would be certain times without the existence of `00:00` data. In such cases, if there might be a need to manually delete the remaining files, and keep the first one alone.

NOTE: Each month gives ~6 to 7 GB of data - thus, make sure there is good internet connectivity and memory for storage. To remedy this, we provide the dataset folds in the google drive link below (thus you will need to make the cross validation and mask generation yourself). This amounts to ~0.3 GB per fold. 
#### Data processing
With the data arranged in the prescribed format, the next step is data processing, which is performed in `DataProcessing/` folder. The AIA preparation and OMNI preparation are outlined in the corresponding folder.

#### Benchmark modelling
If the cross validation dataset generation is done correctly, the Benchmark modelling is quite trivial. Each script runs different Benchmark model (except the 27-day persistence), and is outlined in `BenchmarkModels/`

#### WindNet modelling
Similar to Benchmark modelling, this is trivial to perform if dataset generation is done correctly. There are Jupyter notebooks available for visualization too. 
We have provided the trained models in `Models/` folder, but the dataset will need to be downloaded for visualization. To download all the model files and the GoogleNet weights (to be present in KaffeModelConverted/), please navigate here: https://drive.google.com/drive/folders/18Z9-IcjtUOdaYWOQvPnB8YCKZ4kTj581?usp=sharing. 

NOTE: The set of models for all combinations comes to ~ 100 GB. Thus, as a first step, we advise you to download the best models for 193 $\AA$ and 211 $\AA$ 4 day history and 3 day delay to obtain at least the prediction plots. 

#### Visualization and plotting
1. Each WindNet model may be visualized through `WindNet_easy_crossValid.ipynb` - this also generates the prediction plots of our paper.
2. Grad-CAMS may be generated for the training set, and then the combination for CH/AR performed. This is outlined in `WindNet/README.md`.
3. For generating plots for the paper, run the `MetricMaker.ipynb` notebook. Also, some of the plots are present in `MakePlots.ipynb`. However, please go through `WindNet/README.md` first. 

------------------------------------------

#### Requirements

Code is written on `python 2.7`, and requires:

1. networkx==2.2
2. matplotlib=2.2.4
3. numpy==1.16.5
4. PyWavelets==1.0.3
5. scipy==1.2.2
6. tensorflow==1.7.0
7. cv2==4.1.0
8. skimage==0.14.2
9. sklearn==0.20.3
