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
// FaceTrackingVid.cpp : Defines the entry point for the console application for tracking faces in videos.

// Libraries for landmark detection (includes CLNF and CLM modules)
#include "LandmarkCoreIncludes.h"
#include "GazeEstimation.h"

#include <fstream>
#include <sstream>

// OpenCV includes
#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <SequenceCapture.h>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

static void printErrorAndAbort( const std::string & error )
{
    std::cout << error << std::endl;
    abort();
}

#define FATAL_STREAM( stream ) \
printErrorAndAbort( std::string( "Fatal error: " ) + stream )

using namespace std;

vector<string> get_arguments(int argc, char **argv)
{

	vector<string> arguments;

	for(int i = 0; i < argc; ++i)
	{
		arguments.push_back(string(argv[i]));
	}
	return arguments;
}

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;
int frame_count = 0;

// Visualising the results, TODO move to separate
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
		LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar(vis_certainty*255.0, 0, (1 - vis_certainty) * 255), thickness, fx, fy, cx, cy);

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

	LandmarkDetector::FaceModelParameters det_parameters(arguments);
	det_parameters.track_gaze = true;

	// The modules that are being used for tracking
	LandmarkDetector::CLNF clnf_model(det_parameters.model_location);	

	// Open a sequence
	Utilities::SequenceCapture sequence_reader;

	while (true) // this is not a for loop as we might also be reading from a webcam
	{

		// The sequence reader chooses what to open based on command line arguments provided
		bool reader_opened = sequence_reader.Open(arguments);

		if(!reader_opened && sequence_reader.no_input_specified)
		{
			// If that fails, revert to webcam
			INFO_STREAM("No input specified, attempting to open a webcam 0");
			if (!sequence_reader.OpenWebcam(0))
				ERROR_STREAM("Failed to open the webcam");
		}
		else
		{
			ERROR_STREAM("Failed to open a sequence");
			break;
		}

		INFO_STREAM("Device or file opened");

		cv::Mat captured_image;

		cv::Mat captured_image;
		captured_image = sequence_reader.GetNextFrame();

		INFO_STREAM("Starting tracking");
		while (!captured_image.empty()) // this is not a for loop as we might also be reading from a webcam
		{

			// Reading the images
			cv::Mat_<uchar> grayscale_image;

			// The actual facial landmark detection / tracking
			bool detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, clnf_model, det_parameters);

			// Visualising the results
			// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
			double detection_certainty = clnf_model.detection_certainty;

			// Gaze tracking, absolute gaze direction
			cv::Point3f gazeDirection0(0, 0, -1);
			cv::Point3f gazeDirection1(0, 0, -1);

			if (det_parameters.track_gaze && detection_success && clnf_model.eye_model)
			{
				GazeAnalysis::EstimateGaze(clnf_model, gazeDirection0, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, true);
				GazeAnalysis::EstimateGaze(clnf_model, gazeDirection1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy, false);
			}

			visualise_tracking(captured_image, clnf_model, det_parameters, gazeDirection0, gazeDirection1, sequence_reader.fx, sequence_reader.fy, sequence_reader.cx, sequence_reader.cy);

			// detect key presses
			char character_press = cv::waitKey(1);

			// restart the tracker
			if (character_press == 'r')
			{
				clnf_model.Reset();
			}
			// quit the application
			else if (character_press == 'q')
			{
				return(0);
			}

		}
		
		// Reset the model, for the next video
		clnf_model.Reset();

	}
	return 0;
}

