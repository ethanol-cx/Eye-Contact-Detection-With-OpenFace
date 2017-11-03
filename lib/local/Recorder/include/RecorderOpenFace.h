///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
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

#ifndef __RECORDER_OPENFACE_h_
#define __RECORDER_OPENFACE_h_

#include "RecorderCSV.h"
#include "RecorderHOG.h"

// System includes
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace Recorder
{

	//===========================================================================
	/**
	A class for recording CSV file from OpenFace
	*/
	class RecorderOpenFace {

	public:

		// The constructor for the recorder, need to specify if we are recording a sequence or not
		RecorderOpenFace(const std::string out_directory, const std::string in_filename, bool sequence, bool output_2D_landmarks, bool output_3D_landmarks, bool output_model_params, bool output_pose,
			bool output_AUs, bool output_gaze, bool output_hog, bool output_tracked_video, bool output_aligned_faces, int num_face_landmarks, int num_model_modes, int num_eye_landmarks, 
			const std::vector<std::string>& au_names_class, const std::vector<std::string>& au_names_reg, const std::string& output_codec, double fps_vid_in);

		// Simplified constructor that records all, TODO implement
		RecorderOpenFace(const std::string out_directory, const std::string in_filename, bool sequence, int num_face_landmarks, int num_model_modes, int num_eye_landmarks,
			const std::vector<std::string>& au_names_class, const std::vector<std::string>& au_names_reg);

		// TODO copy, move, destructors

		// Closing and cleaning up the recorder
		void Close();

		// Adding observations to the recorder

		// Required observations for video/image-sequence
		void SetObservationTimestamp(double timestamp);

		// All observations relevant to facial landmarks
		void SetObservationLandmarks(const cv::Mat_<double>& landmarks_2D, const cv::Mat_<double>& landmarks_3D, 
			const cv::Vec6d& params_global, const cv::Mat_<double>& params_local, double confidence, bool success);

		// Pose related observations
		void SetObservationPose(const cv::Vec6d& pose);

		// AU related observations
		void SetObservationActionUnits(const std::vector<std::pair<std::string, double> >& au_intensities, 
			const std::vector<std::pair<std::string, double> >& au_occurences);

		// Gaze related observations
		void SetObservationGaze(const cv::Point3f& gazeDirection0, const cv::Point3f& gazeDirection1,
			const cv::Vec2d& gaze_angle, const std::vector<cv::Point2d>& eye_landmarks);

		// Face alignment related observations
		void SetObservationFaceAlign(const cv::Mat& aligned_face);

		// HOG feature related observations
		void SetObservationHOG(bool good_frame, const cv::Mat_<double>& hog_descriptor, int num_cols, int num_rows, int num_channels);

		void SetObservationVisualization(const cv::Mat_<double> &vis_track);

		void WriteObservation();

	private:

		// Keep track of the file and output root location
		std::string record_root;
		std::string filename;

		// The actual output file stream that will be written
		RecorderCSV csv_recorder;
		RecorderHOG hog_recorder;

		// If we are recording results from a sequence each row refers to a frame, if we are recording an image each row is a face
		bool is_sequence;
		
		// Keep track of what we are recording
		bool output_2D_landmarks;
		bool output_3D_landmarks;
		bool output_model_params;
		bool output_pose;
		bool output_AUs;
		bool output_gaze;
		bool output_hog;
		bool output_tracked_video;
		bool output_aligned_faces;

		// The actual temporary storage for the observations
		double timestamp;

		// Facial landmark related observations
		cv::Mat_<double> landmarks_2D;
		cv::Mat_<double> landmarks_3D;
		cv::Vec6d pdm_params_global;
		cv::Mat_<double> pdm_params_local;
		double landmark_detection_confidence;
		bool landmark_detection_success;

		// Head pose related observations
		cv::Vec6d head_pose;

		// Action Unit related observations
		std::vector<std::pair<std::string, double> > au_intensities;
		std::vector<std::pair<std::string, double> > au_occurences;

		// Gaze related observations
		cv::Point3f gaze_direction0;
		cv::Point3f gaze_direction1;
		cv::Vec2d gaze_angle;
		std::vector<cv::Point2d> eye_landmarks;

		int observation_count;

		// For video writing
		cv::VideoWriter video_writer;
		std::string video_filename;
		std::string output_codec;
		double fps_vid_out;
		cv::Mat_<double> vis_to_out;

	};
}
#endif