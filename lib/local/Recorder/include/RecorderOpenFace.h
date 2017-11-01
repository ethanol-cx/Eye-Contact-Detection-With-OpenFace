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
#include <opencv2/core/core.hpp>

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
			const std::vector<std::string>& au_names_class, const std::vector<std::string>& au_names_reg);

		// Simplified constructor that records all
		RecorderOpenFace(const std::string out_directory, const std::string in_filename, bool sequence, int num_face_landmarks, int num_model_modes, int num_eye_landmarks,
			const std::vector<std::string>& au_names_class, const std::vector<std::string>& au_names_reg);

		// TODO copy, move, destructors

		// Closing and cleaning up the recorder
		void Close();

		// Adding observations to the recorder
		void AddObservationLandmarks(const cv::Mat_<double>& landmarks_2D, const cv::Mat_<double>& landmarks_3D);
		void AddObservationLandmarkParameters(const cv::Vec6d& params_global, const cv::Mat_<double>& params_local);
		void AddObservationPose(const cv::Vec6d& pose);
		void AddObservationActionUnits(const std::vector<std::pair<std::string, double> >& au_intensities, 
			const std::vector<std::pair<std::string, double> >& au_occurences);
		void AddObservationGaze(const cv::Point3f& gazeDirection0, const cv::Point3f& gazeDirection1, 
			const cv::Vec2d& gaze_angle, const cv::Mat_<double>& eye_landmarks);
		void AddObservationFaceAlign(const cv::Mat& aligned_face);
		void AddObservationHOG(bool good_frame, const cv::Mat_<double>& hog_descriptor, int num_cols, int num_rows, int num_channels);
		void AddObservationSuccess(double confidence, bool success);
		void AddObservationTimestamp(double timestamp);

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

	};
}
#endif