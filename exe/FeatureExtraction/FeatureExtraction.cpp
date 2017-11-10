///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt

//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace: an open source facial behavior analysis toolkit
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency
//       in IEEE Winter Conference on Applications of Computer Vision, 2016  
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-speci?c normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
//       Constrained Local Neural Fields for robust facial landmark detection in the wild.
//       Tadas Baltrušaitis, Peter Robinson, and Louis-Philippe Morency. 
//       in IEEE Int. Conference on Computer Vision Workshops, 300 Faces in-the-Wild Challenge, 2013.    
//
///////////////////////////////////////////////////////////////////////////////


// FeatureExtraction.cpp : Defines the entry point for the feature extraction console application.

// System includes
#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>

// Local includes
#include "LandmarkCoreIncludes.h"

#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>
#include <RecorderOpenFace.h>
#include <RecorderOpenFaceParameters.h>
#include <SequenceCapture.h>

#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	// First argument is reserved for the name of the executable
	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

void get_visualization_params(bool& visualize_track, bool& visualize_align, bool& visualize_hog, vector<string> &arguments);

// Some globals for tracking timing information for visualisation (TODO bit ugly)
double fps_tracker = -1.0;
int64 t0 = 0;
int frame_count = 0;

// Visualising the results TODO separate class
void visualise_tracking(cv::Mat& captured_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, double fx, double fy, double cx, double cy)
{

	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = face_model.detection_certainty;
	bool detection_success = face_model.detection_success;

	double visualisation_boundary = 0.4;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty > visualisation_boundary)
	{
		LandmarkDetector::Draw(captured_image, face_model);

		double vis_certainty = detection_certainty;
		if (vis_certainty > 1)
			vis_certainty = 1;

		// Scale from 0 to 1, to allow to indicated by colour how confident we are in the tracking
		vis_certainty = (vis_certainty - visualisation_boundary) / (1 - visualisation_boundary);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetPose(face_model, fx, fy, cx, cy);

		// Draw it in reddish if uncertain, blueish if certain
		LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar(vis_certainty*255.0, 0, (1-vis_certainty) * 255), thickness, fx, fy, cx, cy);

		if (det_parameters.track_gaze && detection_success && face_model.eye_model)
		{
			GazeAnalysis::DrawGaze(captured_image, face_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
		}
	}

	// Work out the framerate TODO
	if (frame_count % 10 == 0)
	{
		double t1 = cv::getTickCount();
		fps_tracker = 10.0 / (double(t1 - t0) / cv::getTickFrequency());
		t0 = t1;
	}

	// Write out the framerate on the image before displaying it
	char fpsC[255];
	std::sprintf(fpsC, "%d", (int)fps_tracker);
	string fpsSt("FPS:");
	fpsSt += fpsC;
	cv::putText(captured_image, fpsSt, cv::Point(10, 20), CV_FONT_HERSHEY_SIMPLEX, 0.5, CV_RGB(255, 0, 0), 1, CV_AA);
	frame_count++;

}

int main (int argc, char **argv)
{

	vector<string> arguments = get_arguments(argc, argv);

	// Deciding what to visualize
	bool visualize_track = false;
	bool visualize_align = false;
	bool visualize_hog = false;
	get_visualization_params(visualize_track, visualize_align, visualize_hog, arguments);

	// Load the modules that are being used for tracking and face analysis
	// Load face landmark detector
	LandmarkDetector::FaceModelParameters det_parameters(arguments);
	// Always track gaze in feature extraction
	det_parameters.track_gaze = true;
	LandmarkDetector::CLNF face_model(det_parameters.model_location);

	// Load facial feature extractor and AU analyser
	FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
	FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

	Utilities::SequenceCapture sequence_reader;

	while (true) // this is not a for loop as we might also be reading from a webcam
	{

		// The sequence reader chooses what to open based on command line arguments provided
		if(!sequence_reader.Open(arguments))
			break;

		INFO_STREAM("Device or file opened");
		
		cv::Mat captured_image;
		
		Utilities::RecorderOpenFaceParameters recording_params(arguments, true, sequence_reader.fps);
		Utilities::RecorderOpenFace open_face_rec(sequence_reader.name, recording_params, arguments);

		captured_image = sequence_reader.GetNextFrame();

		// For reporting progress
		double reported_completion = 0;

		INFO_STREAM("Starting tracking");
		while (!captured_image.empty())
		{

			// Converting to grayscale
			cv::Mat_<uchar> grayscale_image = sequence_reader.GetGrayFrame();

			// The actual facial landmark detection / tracking
			bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_model, det_parameters);

			// Gaze tracking, absolute gaze direction
			cv::Point3f gazeDirection0(0, 0, -1);
			cv::Point3f gazeDirection1(0, 0, -1);
			cv::Vec2d gazeAngle(0, 0);

			if (det_parameters.track_gaze && detection_success && face_model.eye_model)
			{
				GazeAnalysis::EstimateGaze(face_model, gazeDirection0, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true);
				GazeAnalysis::EstimateGaze(face_model, gazeDirection1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, false);
				gazeAngle = GazeAnalysis::GetGazeAngle(gazeDirection0, gazeDirection1);
			}

			// Do face alignment
			cv::Mat sim_warped_img;
			cv::Mat_<double> hog_descriptor;
			int num_hog_rows = 0, num_hog_cols = 0;

			// As this can be expensive only compute it if needed by output or visualization
			if (recording_params.outputAlignedFaces() || recording_params.outputHOG() || recording_params.outputAUs() || visualize_align || visualize_hog)
			{
				face_analyser.AddNextFrame(captured_image, face_model.detected_landmarks, face_model.detection_success, sequence_reader.time_stamp, false, !det_parameters.quiet_mode);
				face_analyser.GetLatestAlignedFace(sim_warped_img);

				if (!det_parameters.quiet_mode && visualize_align)
				{
					cv::imshow("sim_warp", sim_warped_img);
				}
				if (recording_params.outputHOG() || (visualize_hog && !det_parameters.quiet_mode))
				{
					face_analyser.GetLatestHOG(hog_descriptor, num_hog_rows, num_hog_cols);

					if (visualize_hog && !det_parameters.quiet_mode)
					{
						cv::Mat_<double> hog_descriptor_vis;
						FaceAnalysis::Visualise_FHOG(hog_descriptor, num_hog_rows, num_hog_cols, hog_descriptor_vis);
						cv::imshow("hog", hog_descriptor_vis);
					}
				}
			}

			// Work out the pose of the head from the tracked model
			cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);

			// Drawing the visualization on the captured image
			if (recording_params.outputTrackedVideo() || (visualize_track && !det_parameters.quiet_mode))
			{
				visualise_tracking(captured_image, face_model, det_parameters, gazeDirection0, gazeDirection1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);
			}

			// Setting up the recorder output
			open_face_rec.SetObservationHOG(detection_success, hog_descriptor, num_hog_rows, num_hog_cols, 31); // The number of channels in HOG is fixed at the moment, as using FHOG
			open_face_rec.SetObservationVisualization(captured_image);
			open_face_rec.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());
			open_face_rec.SetObservationGaze(gazeDirection0, gazeDirection1, gazeAngle, LandmarkDetector::CalculateAllEyeLandmarks(face_model));
			open_face_rec.SetObservationLandmarks(face_model.detected_landmarks, face_model.GetShape(sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy), 
				face_model.params_global, face_model.params_local, face_model.detection_certainty, detection_success);
			open_face_rec.SetObservationPose(pose_estimate);
			open_face_rec.SetObservationTimestamp(sequence_reader.time_stamp);
			open_face_rec.SetObservationFaceAlign(sim_warped_img);
			open_face_rec.WriteObservation();

			// Visualize the image if desired
			if (visualize_track && !det_parameters.quiet_mode)
			{
				cv::namedWindow("tracking_result", 1);
				cv::imshow("tracking_result", captured_image);
				cv::waitKey(1);
			}

			// Grabbing the next frame (todo this should be part of capture)
			captured_image = sequence_reader.GetNextFrame();

			// Reporting progress
			if(sequence_reader.GetProgress() >= reported_completion / 10.0)
			{
				cout << reported_completion * 10 << "% ";
				reported_completion = reported_completion + 1;
			}

		}
		
		open_face_rec.Close();

		if (recording_params.outputAUs())
		{
			cout << "Postprocessing the Action Unit predictions" << endl;
			face_analyser.PostprocessOutputFile(open_face_rec.GetCSVFile());
		}

		// Reset the models for the next video
		face_analyser.Reset();
		face_model.Reset();

	}

	return 0;
}

void get_visualization_params(bool& visualize_track, bool& visualize_align, bool& visualize_hog,vector<string> &arguments)
{

	bool* valid = new bool[arguments.size()];

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
	}

	string output_root = "";

	visualize_align = false;
	visualize_hog = false;
	visualize_track = false;

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-verbose") == 0)
		{
			visualize_track = true;
			visualize_align = true;
			visualize_hog = true;
		}
		else if (arguments[i].compare("-vis-align") == 0)
		{
			visualize_align = true;
			valid[i] = false;
		}
		else if (arguments[i].compare("-vis-hog") == 0)
		{
			visualize_hog = true;
			valid[i] = false;
		}
		else if (arguments[i].compare("-vis-track") == 0)
		{
			visualize_track = true;
			valid[i] = false;
		}
	}

	for (int i = arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}

}

