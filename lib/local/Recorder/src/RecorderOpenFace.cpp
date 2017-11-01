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

#include "RecorderOpenFace.h"

// For sorting
#include <algorithm>

// File manipulation
#include <fstream>
#include <sstream>
#include <iostream>

// Boost includes for file system manipulation
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>
#include <boost/algorithm/string.hpp>

using namespace boost::filesystem;

using namespace Recorder;

void create_directory(std::string output_path)
{

	// Creating the right directory structure
	auto p = path(output_path);

	if (!boost::filesystem::exists(p))
	{
		bool success = boost::filesystem::create_directories(p);

		if (!success)
		{
			std::cout << "Failed to create a directory..." << p.string() << std::endl;
		}
	}
}


RecorderOpenFace::RecorderOpenFace(const std::string out_directory, const std::string in_filename, bool sequence, bool output_2D_landmarks, bool output_3D_landmarks, bool output_model_params, bool output_pose,
	bool output_AUs, bool output_gaze, bool output_hog, bool output_tracked_video, bool output_aligned_faces, int num_face_landmarks, int num_model_modes, int num_eye_landmarks,
	const std::vector<std::string>& au_names_class, const std::vector<std::string>& au_names_reg):
	is_sequence(sequence), output_2D_landmarks(output_2D_landmarks), output_3D_landmarks(output_3D_landmarks), output_aligned_faces(output_aligned_faces),
	output_AUs(output_AUs), output_gaze(output_gaze), output_hog(output_hog), output_model_params(output_model_params),
	output_pose(output_pose), output_tracked_video(output_tracked_video)
{

	// From the filename, strip out the name without directory and extension
	filename = path(in_filename).replace_extension("").filename().string();
	record_root = out_directory;

	// Construct the directories required for the output
	create_directory(record_root);

	// Create the required individual recorders, CSV, HOG, aligned, video
	std::string csv_filename = (path(record_root) / path(filename).replace_extension(".csv")).string();
	csv_recorder.open(csv_filename, output_2D_landmarks, output_3D_landmarks, output_model_params, output_pose,
		output_AUs, output_gaze, num_face_landmarks, num_model_modes, num_eye_landmarks, au_names_class, au_names_reg);

	// Consruct HOG recorder here
	if(output_hog)
	{
		std::string hog_filename = (path(record_root) / path(filename).replace_extension(".hog")).string();
		hog_recorder.Open(hog_filename);
	}

	// TODO construct a video recorder

	// Prepare image recording

}

void RecorderOpenFace::AddObservationHOG(bool good_frame, const cv::Mat_<double>& hog_descriptor, int num_cols, int num_rows, int num_channels)
{
	hog_recorder.AddObservationHOG(good_frame, hog_descriptor, num_cols, num_rows, num_channels);
}

// TODO the other 4 constructors + destructors?

