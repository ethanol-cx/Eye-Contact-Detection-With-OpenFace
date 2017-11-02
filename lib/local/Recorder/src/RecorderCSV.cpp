///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis, all rights reserved.
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

#include "RecorderCSV.h"

// For sorting
#include <algorithm>

using namespace Recorder;

// Default constructor initializes the variables
RecorderCSV::RecorderCSV():output_file(){};

// TODO the other 4 constructors + destructors?

// Opening the file and preparing the header for it
bool RecorderCSV::Open(std::string output_file_name, bool output_2D_landmarks, bool output_3D_landmarks, bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
	int num_face_landmarks, int num_model_modes, int num_eye_landmarks, std::vector<std::string> au_names_class, std::vector<std::string> au_names_reg)
{
	output_file.open(output_file_name, std::ios_base::out);

	if (!output_file.is_open())
		return false;

	// Different headers if we are writing out the results on a sequence or an individual image
	if(this->is_sequence)
	{
		output_file << "frame, timestamp, confidence, success";
	}
	else
	{
		output_file << "face, confidence";
	}

	if (output_gaze)
	{
		output_file << ", gaze_0_x, gaze_0_y, gaze_0_z, gaze_1_x, gaze_1_y, gaze_1_z, gaze_angle_x, gaze_angle_y";

		for (int i = 0; i < num_eye_landmarks; ++i)
		{
			output_file << ", eye_lmk_x_" << i;
		}
		for (int i = 0; i < num_eye_landmarks; ++i)
		{
			output_file << ", eye_lmk_y_" << i;
		}
	}

	if (output_pose)
	{
		output_file << ", pose_Tx, pose_Ty, pose_Tz, pose_Rx, pose_Ry, pose_Rz";
	}

	if (output_2D_landmarks)
	{
		for (int i = 0; i < num_face_landmarks; ++i)
		{
			output_file << ", x_" << i;
		}
		for (int i = 0; i < num_face_landmarks; ++i)
		{
			output_file << ", y_" << i;
		}
	}

	if (output_3D_landmarks)
	{
		for (int i = 0; i < num_face_landmarks; ++i)
		{
			output_file << ", X_" << i;
		}
		for (int i = 0; i < num_face_landmarks; ++i)
		{
			output_file << ", Y_" << i;
		}
		for (int i = 0; i < num_face_landmarks; ++i)
		{
			output_file << ", Z_" << i;
		}
	}

	// Outputting model parameters (rigid and non-rigid), the first parameters are the 6 rigid shape parameters, they are followed by the non rigid shape parameters
	if (output_model_params)
	{
		output_file << ", p_scale, p_rx, p_ry, p_rz, p_tx, p_ty";
		for (int i = 0; i < num_model_modes; ++i)
		{
			output_file << ", p_" << i;
		}
	}

	if (output_AUs)
	{
		std::sort(au_names_reg.begin(), au_names_reg.end());
		for (std::string reg_name : au_names_reg)
		{
			output_file << ", " << reg_name << "_r";
		}

		std::sort(au_names_class.begin(), au_names_class.end());
		for (std::string class_name : au_names_class)
		{
			output_file << ", " << class_name << "_c";
		}
	}

	output_file << std::endl;

	return true;

}

// TODO check if the stream is open
//void writeLine(int frame_count, double time_stamp, bool detection_success,
//	cv::Point3f gazeDirection0, cv::Point3f gazeDirection1, cv::Vec2d gaze_angle, cv::Vec6d& pose_estimate, double fx, double fy, double cx, double cy,
//	const FaceAnalysis::FaceAnalyser& face_analyser);

// Closing the file and cleaning up
void RecorderCSV::Close()
{
	output_file.close();
}
