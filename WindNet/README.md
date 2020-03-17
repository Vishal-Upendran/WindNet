#### WindNet modelling

The primary script for training is `TrainingScript_cv.py`. This loads in the data, and trains the model. For restoring the model, one must use `ModelRestoration_cv.py`. <br> For a complete tutorial, look at `WindNet_easy_crossValid.ipynb`.

#### Further calculation
Once the models are trained, to generate Grad-cam maps/values over the entire training set, run `Average_GC_generator.py` with appropriate arguments. This saves the values in the respective model folder. <br>
Then, one must obtain the Grad-cam values per CH/AR area. This is done in `GenerateActivationMaps_CV_Predone.ipynb`.<br>
Now, to obtain the final plots, refer to `'../MakePlots.ipynb`. <br>
To generate the tables, please refere to `MetricMaker.ipynb`.
