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

using namespace boost::filesystem;

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

void get_image_input_output_params_feats(vector<vector<string> > &input_image_files, bool& as_video, vector<string> &arguments);

// Some globals for tracking timing information for visualisation
double fps_tracker = -1.0;
int64 t0 = 0;

// Visualising the results
void visualise_tracking(cv::Mat& captured_image, const LandmarkDetector::CLNF& face_model, const LandmarkDetector::FaceModelParameters& det_parameters, cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, int frame_count, double fx, double fy, double cx, double cy)
{

	// Drawing the facial landmarks on the face and the bounding box around it if tracking is successful and initialised
	double detection_certainty = face_model.detection_certainty;
	bool detection_success = face_model.detection_success;

	double visualisation_boundary = 0.2;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (detection_certainty < visualisation_boundary)
	{
		LandmarkDetector::Draw(captured_image, face_model);

		double vis_certainty = detection_certainty;
		if (vis_certainty > 1)
			vis_certainty = 1;
		if (vis_certainty < -1)
			vis_certainty = -1;

		vis_certainty = (vis_certainty + 1) / (visualisation_boundary + 1);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		cv::Vec6d pose_estimate_to_draw = LandmarkDetector::GetPose(face_model, fx, fy, cx, cy);

		// Draw it in reddish if uncertain, blueish if certain
		LandmarkDetector::DrawBox(captured_image, pose_estimate_to_draw, cv::Scalar((1 - vis_certainty)*255.0, 0, vis_certainty * 255), thickness, fx, fy, cx, cy);

		if (det_parameters.track_gaze && detection_success && face_model.eye_model)
		{
			GazeAnalysis::DrawGaze(captured_image, face_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
		}
	}

	// Work out the framerate
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

}

int main (int argc, char **argv)
{

	vector<string> arguments = get_arguments(argc, argv);
	
	// Get the input output file parameters
	vector<string> input_files, output_files;
	string output_codec;
	LandmarkDetector::get_video_input_output_params(input_files, output_files, output_codec, arguments);

	// TODO remove, when have a capture class
	bool video_input = true;
	bool images_as_video = false;

	vector<vector<string> > input_image_files;

	// Adding image support (TODO should be moved to capture)
	if(input_files.empty())
	{
		vector<string> o_img;
		get_image_input_output_params_feats(input_image_files, images_as_video, arguments);	

		if(!input_image_files.empty())
		{
			video_input = false;
		}
	}

	// Grab camera parameters, if they are not defined (approximate values will be used)
	float fx = 0, fy = 0, cx = 0, cy = 0;
	int d = 0;
	// Get camera parameters
	LandmarkDetector::get_camera_params(d, fx, fy, cx, cy, arguments);    
	
	// If cx (optical axis centre) is undefined will use the image size/2 as an estimate
	bool cx_undefined = false;
	bool fx_undefined = false;
	if (cx == 0 || cy == 0)
	{
		cx_undefined = true;
	}
	if (fx == 0 || fy == 0)
	{
		fx_undefined = true;
	}

	// Deciding what to visualize
	bool visualize_track = false;
	bool visualize_align = false;
	bool visualize_hog = false;
	get_visualization_params(visualize_track, visualize_align, visualize_hog, arguments);

	// If multiple video files are tracked, use this to indicate if we are done
	bool done = false;	
	int f_n = -1;
	int curr_img = -1;

	// Load the modules that are being used for tracking and face analysis
	// Load face landmark detector
	LandmarkDetector::FaceModelParameters det_parameters(arguments);
	// Always track gaze in feature extraction
	det_parameters.track_gaze = true;
	LandmarkDetector::CLNF face_model(det_parameters.model_location);

	// Load facial feature extractor and AU analyser
	FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
	FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

	while (!done) // this is not a for loop as we might also be reading from a webcam
	{

		string current_file;

		cv::VideoCapture video_capture;

		cv::Mat captured_image;
		int total_frames = -1;
		int reported_completion = 0;

		double fps_vid_in = -1.0;

		// TODO this should be moved to a SequenceCapture class
		if (video_input)
		{
			// We might specify multiple video files as arguments
			if (input_files.size() > 0)
			{
				f_n++;
				current_file = input_files[f_n];
			}
			else
			{
				// If we want to write out from webcam
				f_n = 0;
			}
			// Do some grabbing
			if (current_file.size() > 0)
			{
				INFO_STREAM("Attempting to read from file: " << current_file);
				video_capture = cv::VideoCapture(current_file);
				total_frames = (int)video_capture.get(CV_CAP_PROP_FRAME_COUNT);
				fps_vid_in = video_capture.get(CV_CAP_PROP_FPS);

				// Check if fps is nan or less than 0
				if (fps_vid_in != fps_vid_in || fps_vid_in <= 0)
				{
					INFO_STREAM("FPS of the video file cannot be determined, assuming 30");
					fps_vid_in = 30;
				}
			}

			if (!video_capture.isOpened())
			{
				FATAL_STREAM("Failed to open video source, exiting");
				return 1;
			}
			else
			{
				INFO_STREAM("Device or file opened");
			}

			video_capture >> captured_image;
		}
		else
		{
			f_n++;
			curr_img++;
			if (!input_image_files[f_n].empty())
			{
				string curr_img_file = input_image_files[f_n][curr_img];
				captured_image = cv::imread(curr_img_file, -1);
			}
			else
			{
				FATAL_STREAM("No .jpg or .png images in a specified drectory, exiting");
				return 1;
			}

			// If image sequence provided, assume the fps is 30
			fps_vid_in = 30;
		}

		// If optical centers are not defined just use center of image
		if (cx_undefined)
		{
			cx = captured_image.cols / 2.0f;
			cy = captured_image.rows / 2.0f;
		}
		// Use a rough guess-timate of focal length
		if (fx_undefined)
		{
			fx = 500 * (captured_image.cols / 640.0);
			fy = 500 * (captured_image.rows / 480.0);

			fx = (fx + fy) / 2.0;
			fy = fx;
		}

		Utilities::RecorderOpenFaceParameters recording_params(arguments, true, fps_vid_in);
		Utilities::RecorderOpenFace open_face_rec(output_files[f_n], input_files[f_n], recording_params);

		int frame_count = 0;

		// Use for timestamping if using a webcam
		int64 t_initial = cv::getTickCount();

		// Timestamp in seconds of current processing
		double time_stamp = 0;

		INFO_STREAM("Starting tracking");
		while (!captured_image.empty())
		{

			// Grab the timestamp first
			if (video_input)
			{
				time_stamp = (double)frame_count * (1.0 / fps_vid_in);
			}
			else
			{
				// if loading images assume 30fps
				time_stamp = (double)frame_count * (1.0 / 30.0);
			}

			// Reading the images
			cv::Mat_<uchar> grayscale_image;

			if (captured_image.channels() == 3)
			{
				cvtColor(captured_image, grayscale_image, CV_BGR2GRAY);
			}
			else
			{
				grayscale_image = captured_image.clone();
			}

			// The actual facial landmark detection / tracking
			bool detection_success;

			if (video_input || images_as_video)
			{
				detection_success = LandmarkDetector::DetectLandmarksInVideo(grayscale_image, face_model, det_parameters);
			}
			else
			{
				detection_success = LandmarkDetector::DetectLandmarksInImage(grayscale_image, face_model, det_parameters);
			}

			// Gaze tracking, absolute gaze direction
			cv::Point3f gazeDirection0(0, 0, -1);
			cv::Point3f gazeDirection1(0, 0, -1);
			cv::Vec2d gazeAngle(0, 0);

			if (det_parameters.track_gaze && detection_success && face_model.eye_model)
			{
				GazeAnalysis::EstimateGaze(face_model, gazeDirection0, fx, fy, cx, cy, true);
				GazeAnalysis::EstimateGaze(face_model, gazeDirection1, fx, fy, cx, cy, false);
				gazeAngle = GazeAnalysis::GetGazeAngle(gazeDirection0, gazeDirection1);
			}

			// Do face alignment
			cv::Mat sim_warped_img;
			cv::Mat_<double> hog_descriptor;
			int num_hog_rows = 0, num_hog_cols = 0;

			// As this can be expensive only compute it if needed by output or visualization
			if (recording_params.outputAlignedFaces() || recording_params.outputHOG() || recording_params.outputAUs() || visualize_align || visualize_hog)
			{
				face_analyser.AddNextFrame(captured_image, face_model.detected_landmarks, face_model.detection_success, time_stamp, false, !det_parameters.quiet_mode);
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
			cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, fx, fy, cx, cy);

			// Drawing the visualization on the captured image
			if (recording_params.outputTrackedVideo() || (visualize_track && !det_parameters.quiet_mode))
			{
				visualise_tracking(captured_image, face_model, det_parameters, gazeDirection0, gazeDirection1, frame_count, fx, fy, cx, cy);
			}

			// Setting up the recorder output
			open_face_rec.SetObservationHOG(detection_success, hog_descriptor, num_hog_rows, num_hog_cols, 31); // The number of channels in HOG is fixed at the moment, as using FHOG
			open_face_rec.SetObservationVisualization(captured_image);
			open_face_rec.SetObservationActionUnits(face_analyser.GetCurrentAUsReg(), face_analyser.GetCurrentAUsClass());
			open_face_rec.SetObservationGaze(gazeDirection0, gazeDirection1, gazeAngle, LandmarkDetector::CalculateAllEyeLandmarks(face_model));
			open_face_rec.SetObservationLandmarks(face_model.detected_landmarks, face_model.GetShape(fx, fy, cx, cy), face_model.params_global, face_model.params_local, face_model.detection_certainty, detection_success);
			open_face_rec.SetObservationPose(pose_estimate);
			open_face_rec.SetObservationTimestamp(time_stamp);
			open_face_rec.WriteObservation();

			// Visualize the image if desired
			if (visualize_track && !det_parameters.quiet_mode)
			{
				cv::namedWindow("tracking_result", 1);
				cv::imshow("tracking_result", captured_image);
			}

			// Grabbing the next frame (todo this should be part of capture)
			if(video_input)
			{
				video_capture >> captured_image;
			}
			else
			{
				curr_img++;
				if(curr_img < (int)input_image_files[f_n].size())
				{
					string curr_img_file = input_image_files[f_n][curr_img];
					captured_image = cv::imread(curr_img_file, -1);
				}
				else
				{
					captured_image = cv::Mat();
				}
			}
			
			if (!det_parameters.quiet_mode)
			{
				// detect key presses
				char character_press = cv::waitKey(1);
			
				// restart the tracker
				if(character_press == 'r')
				{
					face_model.Reset();
				}
				// quit the application
				else if(character_press=='q')
				{
					return(0);
				}
			}
			
			// Update the frame count
			frame_count++;

			if(total_frames != -1)
			{
				if((double)frame_count/(double)total_frames >= reported_completion / 10.0)
				{
					cout << reported_completion * 10 << "% ";
					reported_completion = reported_completion + 1;
				}
			}

		}
		
		open_face_rec.Close();

		if (output_files.size() > 0 && recording_params.outputAUs())
		{
			cout << "Postprocessing the Action Unit predictions" << endl;
			face_analyser.PostprocessOutputFile(open_face_rec.GetCSVFile());
		}

		// Reset the models for the next video
		face_analyser.Reset();
		face_model.Reset();

		frame_count = 0;
		curr_img = -1;

		if (total_frames != -1)
		{
			cout << endl;
		}

		// break out of the loop if done with all the files (or using a webcam)
		if((video_input && f_n == input_files.size() -1) || (!video_input && f_n == input_image_files.size() - 1))
		{
			done = true;
		}
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

// Can process images via directories creating a separate output file per directory
void get_image_input_output_params_feats(vector<vector<string> > &input_image_files, bool& as_video, vector<string> &arguments)
{
	bool* valid = new bool[arguments.size()];

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
		if (arguments[i].compare("-fdir") == 0)
		{

			// parse the -fdir directory by reading in all of the .png and .jpg files in it
			path image_directory(arguments[i + 1]);

			try
			{
				// does the file exist and is it a directory
				if (exists(image_directory) && is_directory(image_directory))
				{

					vector<path> file_in_directory;
					copy(directory_iterator(image_directory), directory_iterator(), back_inserter(file_in_directory));

					// Sort the images in the directory first
					sort(file_in_directory.begin(), file_in_directory.end());

					vector<string> curr_dir_files;

					for (vector<path>::const_iterator file_iterator(file_in_directory.begin()); file_iterator != file_in_directory.end(); ++file_iterator)
					{
						// Possible image extension .jpg and .png
						if (file_iterator->extension().string().compare(".jpg") == 0 || file_iterator->extension().string().compare(".png") == 0)
						{
							curr_dir_files.push_back(file_iterator->string());
						}
					}

					input_image_files.push_back(curr_dir_files);
				}
			}
			catch (const filesystem_error& ex)
			{
				cout << ex.what() << '\n';
			}

			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-asvid") == 0)
		{
			as_video = true;
		}
	}

	// Clear up the argument list
	for (int i = arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}

}