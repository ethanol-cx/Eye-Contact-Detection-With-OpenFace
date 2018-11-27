# Eye Contact Detection Project – Python Appendix
Here's our suite of tools for creating models to predict eye contact with OpenFace features
To create models to use in the eye contact prediction software, follow these steps:

1. Use OpenFace Feature extraction on training data videos to obtain the features csv. Annotate the video with ELAN for when the subject is looking or not. Export these annotations as a csv.

2. Combine feature extractions and annotations with processor.py. Follow the instructions within the file for specifying the csv files.

3. Use stacker.py to combine individual videos into one large dataset before training.

4. Use any of the machine learning files (gaussianTrain.py, ) to train and output a .joblib file of your predictive model.

5. Specify the .joblib file in the predictorServer.py for the server to make predictions from.

##### Note – the features in openface you are using to predict must match the features the .joblib file is trained from. The same number of features but not the same will result in inaccurate predictions, and using a different number of features will cause errors.

## Eye Contact Detection Files and Descriptions

`processor.py` - Used for matching ELAN annotations in csv form with OpenFace features after performing feature extraction on a video.
Specify the annotations file name in line 2, the openface features file name in line 10, and the name of your combined output csv file in line 3.

`stacker.py` - Used for stacking combined csvs to have all of our training data in one csv file.
Specify the first file in line 3, and the file to add to it in line 4.

`predictorServer.py` - The predictorServer file is used for calling predictions in real time from OpenFace.
Specify the model file you'd like to call predictions from in line 18 as a joblib file.

`predictorClient.py` - The client file is used to connect to the server and ask for a prediction from command line arguments. There is additional commented out functionality within this file that you can un-comment to allow for real-time `csv` prediction output as well as checking against annotations while running on a pre-recorded/pre-annotated video.

`gaussianTrain.py` - Used for training, evaluating and serializing as a joblib a Gaussian Naive Bayes model on the data.
Specify the data set file name you'd like to train from in line 10.

`.joblib` files - Serialized already trained models to be used for calling predictions

`CombinedCsvs` folder - Annotations and Feature extractions for individual videos we're training withs
