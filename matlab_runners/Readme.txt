--------------------------------------- OpenFace Matlab runners -----------------------------------------------------------------------------	

These are provided for recreation of some of the experiments described in the publications and to demonstrate the command line interface by calling the C++ executables from Matlab.

======================== Demos ==================================

run_demo_images.m - running the FaceLandmarkImg landmark detection on the demo images packaged with the code
run_demo_videos.m - running the FaceTrackingVid landmark detection and tracking on prepackaged demo videos
run_demo_video_multi.m - running the FaceTrackingVidMulti landmark detection and tracking on prepackaged demo videos (the difference from above is that it can deal with multiple faces)

For extracting head pose, facial landmarks, HOG features, aligned faces, eye gaze, and Facial Action Units look at the following demos:
	feature_extraction_demo_img_seq.m - Running the FeatureExtraction project, it demonstrates how to specify parameters for extracting a number of features from a sequence of images in a folder and how to read those features into Matlab.	
	feature_extraction_demo_vid.m - Running the FeatureExtraction project, it demonstrates how to specify parameters for extracting a number of features from a video and how to read those features into Matlab.	
	gaze_extraction_demo_vid.m - Example of a clip with varying gaze and extraction of eye gaze information

The other scripts are for unit testing of the code:
- run_demo_align_size.m
- run_tes_img_seq.m
	
The demos are configured to use CLNF patch experts trained on in-the-wild and Multi-PIE datasets, it is possible to uncomment other model file definitions in the scripts to run them instead.

======================== Head Pose Experiments ============================
To run them you will need to have the appropriate datasets and to change the dataset locations.

run_head_pose_tests_OpenFace.m - runs CLNF on the 3 head pose datasets (Boston University, Biwi Kinect, and ICT-3DHP you need to acquire the datasets yourself)

======================== Feature Point Experiments ============================

run_OpenFace_feature_point_tests_300W.m runs CLM and CLNF on the in the wild face datasets acquired from  http://ibug.doc.ic.ac.uk/resources/300-W/ 
The code uses the already defined bounding boxes of faces (these are produced using the 'ExtractBoundingBoxes.m' script on the in the wild datasets). The code relies on there being a .txt file of the same name as the image containing the bounding box in the appropriate directory, see https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments for details.

To run the code you will need to download the 300-W challenge datasets and then replace the database_root with the dataset location.

This script also includes code to draw a graph displaying error curves of the CLNF and CLM methods trained on in the wild data.

For convenient comparisons to other state-of-art approaches it also includes results of using the following approaches on the 300-W datasets:

run_yt_dataset.m run the CLNF model on the YTCeleb Database (https://sites.google.com/site/akshayasthana/Annotations), you need to get the dataset yourself though.

======================== Action Unit Experiments ============================

Evaluating our Facial Action Unit detection system on Bosphorus, BP4D, DISFA, FERA2011 and SEMAINE datasets.

As the models were partially trained/validated on DISFA, FERA2011, BP4D, UNBC, Bosphorus, and SEMAINE datasets the results might not generalise across datasets. However, this demonstrates how AU prediction can be done with our system.

======================== Gaze Experiments ============================

Evaluating our gaze estimation on the MPIIGaze dataset, run the extract_mpii_gaze_test.m script in the Gaze Experiments folder
