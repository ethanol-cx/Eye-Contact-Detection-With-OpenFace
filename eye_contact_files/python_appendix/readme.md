# Eye Contact Detection Project – Python Appendix
Here's our suite of tools for creating models to predict eye contact with OpenFace features, as well as their descriptions. Please note the commented out code in `predictorClient.py` and uncomment lines for added functionality if desired.

The files here are as follows:

`processor.py` - Used for matching ELAN annotations in csv form with OpenFace features after performing feature extraction on a video.
Specify the annotations file name in line 2, the openface features file name in line 10, and the name of your combined output csv file in line 3.

`stacker.py` - Used for stacking combined csvs to have all of our training data in one csv file.
Specify the first file in line 3, and the file to add to it in line 4.

`predictorServer.py `- The predictorServer file is used for calling predictions in real time from OpenFace.
Specify the model file you'd like to call predictions from in line 18 as a joblib file.

`predictorClient.py` - The client file is used to connect to the server and ask for a prediction from command line arguments

`gaussianTrain.py` - Used for training, evaluating and serializing as a joblib a Gaussian Naive Bayes model on the data.
Specify the data set file name you'd like to train from in line 10.

`.joblib` files - Serialized already trained models to be used for calling predictions

`CombinedCsvs` folder - Annotations and Feature extractions for individual videos we're training with
