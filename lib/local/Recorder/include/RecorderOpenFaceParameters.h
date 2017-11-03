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

//  Parameters of the Face analyser
#ifndef __RECORDER_OPENFACE_PARAM_H
#define __RECORDER_OPENFACE_PARAM_H

#include <vector>
#include <opencv2/core/core.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

using namespace std;

namespace Recorder
{

	struct RecorderOpenFaceParameters
	{
	public:
		// Constructors
		RecorderOpenFaceParameters();
		RecorderOpenFaceParameters(vector<string> &arguments);



		bool isSequence() const { return is_sequence; }
		bool output2DLandmarks() const { return output_2D_landmarks; }
		bool output3DLandmarks() const { return output_3D_landmarks; }
		bool outputPDMParams() const { return output_model_params; }
		bool outputPose() const { return output_pose; }
		bool outputAUs() const { return output_AUs; }
		bool outputGaze() const { return output_gaze; }
		bool outputHOG() const { return output_hog; }
		bool outputTrackedVideo() const { return output_tracked_video; }
		bool outputAlignedFaces() const { return output_aligned_faces; }

	private:

		// The default values initializer
		void init();

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

#endif // __FACE_ANALYSER_PARAM_H
