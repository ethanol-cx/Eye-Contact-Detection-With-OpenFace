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

#include "Visualizer.h"
#include "VisualizationUtils.h"

using namespace Utilities;

Visualizer::Visualizer(std::vector<std::string> arguments)
{
	// By default not visualizing anything
	this->vis_track = false;
	this->vis_hog = false;
	this->vis_align = false;

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-verbose") == 0)
		{
			vis_track = true;
			vis_align = true;
			vis_hog = true;
		}
		else if (arguments[i].compare("-vis-align") == 0)
		{
			vis_align = true;
		}
		else if (arguments[i].compare("-vis-hog") == 0)
		{
			vis_hog = true;
		}
		else if (arguments[i].compare("-vis-track") == 0)
		{
			vis_track = true;
		}
	}

}

Visualizer::Visualizer(bool vis_track, bool vis_hog, bool vis_align)
{
	this->vis_track = vis_track;
	this->vis_hog = vis_hog;
	this->vis_align = vis_align;
}

void Visualizer::SetImage(const cv::Mat& canvas, float fx, float fy, float cx, float cy)
{
	captured_image = canvas.clone();
	this->fx = fx;
	this->fy = fy;
	this->cx = cx;
	this->cy = cy;
}


void Visualizer::SetObservationFaceAlign(const cv::Mat& aligned_face)
{
	this->aligned_face_image = aligned_face;
}

void Visualizer::SetObservationHOG(const cv::Mat_<double>& hog_descriptor, int num_cols, int num_rows)
{
	 Visualise_FHOG(hog_descriptor, num_rows, num_cols, this->hog_image);
}


void Visualizer::SetObservationLandmarks(const cv::Mat_<double>& landmarks_2D, double confidence, bool success, const cv::Mat_<int>& visibilities)
{
	DrawLandmarkDetResults(captured_image, landmarks_2D, visibilities);
}

void Visualizer::SetObservationPose(const cv::Vec6d& pose, double confidence)
{

	double visualisation_boundary = 0.4;

	// Only draw if the reliability is reasonable, the value is slightly ad-hoc
	if (confidence > visualisation_boundary)
	{
		double vis_certainty = confidence;
		if (vis_certainty > 1)
			vis_certainty = 1;

		// Scale from 0 to 1, to allow to indicated by colour how confident we are in the tracking
		vis_certainty = (vis_certainty - visualisation_boundary) / (1 - visualisation_boundary);

		// A rough heuristic for box around the face width
		int thickness = (int)std::ceil(2.0* ((double)captured_image.cols) / 640.0);

		// Draw it in reddish if uncertain, blueish if certain
		DrawBox(captured_image, pose, cv::Scalar(vis_certainty*255.0, 0, (1 - vis_certainty) * 255), thickness, fx, fy, cx, cy);
	}
}


void Visualizer::SetObservationGaze(const cv::Point3f& gaze_direction0, const cv::Point3f& gaze_direction1,
	const cv::Vec2d& gaze_angle, const std::vector<cv::Point2d>& eye_landmarks)
{
	// TODO actual drawing

	if (det_parameters.track_gaze && detection_success && face_model.eye_model)
	{
		GazeAnalysis::DrawGaze(captured_image, face_model, gazeDirection0, gazeDirection1, fx, fy, cx, cy);
	}
}




