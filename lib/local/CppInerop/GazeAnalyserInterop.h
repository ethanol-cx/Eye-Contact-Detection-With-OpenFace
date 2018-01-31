#pragma once
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

// FaceAnalyser_Interop.h
#ifndef __GAZE_ANALYSER_INTEROP_h_
#define __GAZE_ANALYSER_INTEROP_h_

#pragma once

// Include all the unmanaged things we need.
#pragma managed

#include <msclr\marshal.h>
#include <msclr\marshal_cppstd.h>

#pragma unmanaged

#include <opencv2/core/core.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Allows to overcome boost name clash stuff with C++ CLI
#ifdef __cplusplus_cli
#define generic __identifier(generic)
#endif

#include <OpenCVWrappers.h>
#include <LandmarkDetectorInterop.h>
#include <GazeEstimation.h>

// Boost stuff
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

#ifdef __cplusplus_cli
#undef generic
#endif

#pragma managed

namespace GazeAnalyser_Interop {

	public ref class GazeAnalyserManaged
	{

	private:

		// Variable storing gaze for recording

		// Absolute gaze direction
		cv::Point3f* gazeDirection0;
		cv::Point3f* gazeDirection1;
		cv::Vec2d* gazeAngle;

		cv::Point3f* pupil_left;
		cv::Point3f* pupil_right;

	public:
		GazeAnalyserManaged()
		{
			gazeDirection0 = new cv::Point3f();
			gazeDirection1 = new cv::Point3f();
			gazeAngle = new cv::Vec2d();

			pupil_left = new cv::Point3f();
			pupil_right = new cv::Point3f();
		}

		void AddNextFrame(CppInterop::LandmarkDetector::CLNF^ clnf, bool success, double fx, double fy, double cx, double cy) {

			// After the AUs have been detected do some gaze estimation as well
			GazeAnalysis::EstimateGaze(*clnf->getCLNF(), *gazeDirection0, fx, fy, cx, cy, true);
			GazeAnalysis::EstimateGaze(*clnf->getCLNF(), *gazeDirection1, fx, fy, cx, cy, false);

			// Estimate the gaze angle WRT to head pose here
			System::Collections::Generic::List<double>^ pose_list = gcnew System::Collections::Generic::List<double>();

			*gazeAngle = GazeAnalysis::GetGazeAngle(*gazeDirection0, *gazeDirection1);

			// Grab pupil locations
			int part_left = -1;
			int part_right = -1;
			for (size_t i = 0; i < clnf->getCLNF()->hierarchical_models.size(); ++i)
			{
				if (clnf->getCLNF()->hierarchical_model_names[i].compare("left_eye_28") == 0)
				{
					part_left = i;
				}
				if (clnf->getCLNF()->hierarchical_model_names[i].compare("right_eye_28") == 0)
				{
					part_right = i;
				}
			}

			cv::Mat_<double> eyeLdmks3d_left = clnf->getCLNF()->hierarchical_models[part_left].GetShape(fx, fy, cx, cy);
			cv::Point3f pupil_left_h = GazeAnalysis::GetPupilPosition(eyeLdmks3d_left);
			pupil_left->x = pupil_left_h.x; pupil_left->y = pupil_left_h.y; pupil_left->z = pupil_left_h.z;

			cv::Mat_<double> eyeLdmks3d_right = clnf->getCLNF()->hierarchical_models[part_right].GetShape(fx, fy, cx, cy);
			cv::Point3f pupil_right_h = GazeAnalysis::GetPupilPosition(eyeLdmks3d_right);
			pupil_right->x = pupil_right_h.x; pupil_right->y = pupil_right_h.y; pupil_right->z = pupil_right_h.z;
		}

		System::Tuple<System::Tuple<double, double, double>^, System::Tuple<double, double, double>^>^ GetGazeCamera()
		{

			auto gaze0 = gcnew System::Tuple<double, double, double>(gazeDirection0->x, gazeDirection0->y, gazeDirection0->z);
			auto gaze1 = gcnew System::Tuple<double, double, double>(gazeDirection1->x, gazeDirection1->y, gazeDirection1->z);

			return gcnew System::Tuple<System::Tuple<double, double, double>^, System::Tuple<double, double, double>^>(gaze0, gaze1);

		}

		System::Tuple<double, double>^ GetGazeAngle()
		{
			auto gaze_angle = gcnew System::Tuple<double, double>((*gazeAngle)[0], (*gazeAngle)[1]);
			return gaze_angle;

		}
		System::Collections::Generic::List<System::Tuple<System::Windows::Point, System::Windows::Point>^>^ CalculateGazeLines(float fx, float fy, float cx, float cy)
		{

			cv::Mat_<double> cameraMat = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 0);

			vector<cv::Point3f> points_left;
			points_left.push_back(cv::Point3f(*pupil_left));
			points_left.push_back(cv::Point3f(*pupil_left + *gazeDirection0 * 40.0));

			vector<cv::Point3f> points_right;
			points_right.push_back(cv::Point3f(*pupil_right));
			points_right.push_back(cv::Point3f(*pupil_right + *gazeDirection1 * 40.0));

			// Perform manual projection of points
			vector<cv::Point2d> imagePoints_left;
			for (int i = 0; i < points_left.size(); ++i)
			{
				double x = points_left[i].x * fx / points_left[i].z + cx;
				double y = points_left[i].y * fy / points_left[i].z + cy;
				imagePoints_left.push_back(cv::Point2d(x, y));

			}

			vector<cv::Point2d> imagePoints_right;
			for (int i = 0; i < points_right.size(); ++i)
			{
				double x = points_right[i].x * fx / points_right[i].z + cx;
				double y = points_right[i].y * fy / points_right[i].z + cy;
				imagePoints_right.push_back(cv::Point2d(x, y));

			}

			auto lines = gcnew System::Collections::Generic::List<System::Tuple<System::Windows::Point, System::Windows::Point>^>();
			lines->Add(gcnew System::Tuple<System::Windows::Point, System::Windows::Point>(System::Windows::Point(imagePoints_left[0].x, imagePoints_left[0].y), System::Windows::Point(imagePoints_left[1].x, imagePoints_left[1].y)));
			lines->Add(gcnew System::Tuple<System::Windows::Point, System::Windows::Point>(System::Windows::Point(imagePoints_right[0].x, imagePoints_right[0].y), System::Windows::Point(imagePoints_right[1].x, imagePoints_right[1].y)));
			return lines;
		}

		// Finalizer. Definitely called before Garbage Collection,
		// but not automatically called on explicit Dispose().
		// May be called multiple times.
		!GazeAnalyserManaged()
		{

			delete gazeDirection0;
			delete gazeDirection1;
			delete gazeAngle;

			delete pupil_left;
			delete pupil_right;

		}

		// Destructor. Called on explicit Dispose() only.
		~GazeAnalyserManaged()
		{
			this->!GazeAnalyserManaged();
		}

	};
}

#endif