///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis.
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

#pragma once

#pragma unmanaged

// Include all the unmanaged things we need.

#include "Visualizer.h"

#pragma managed

using System::Collections::Generic::List;

namespace UtilitiesOF {

	public ref class Visualizer
	{
	private:

		// OpenCV based video capture for reading from files
		Utilities::Visualizer* m_visualizer;

	public:

		Visualizer(bool vis_track, bool vis_hog, bool vis_aligned)
		{
			m_visualizer = new Utilities::Visualizer(vis_track, vis_hog, vis_aligned);
		}

		void SetObservationGaze(System::Tuple<double, double, double>^ gaze_direction0, System::Tuple<double, double, double>^ gaze_direction1,
			List<System::Tuple<double, double>^>^ landmarks_2D, List<System::Tuple<double, double, double>^>^ landmarks_3D,
			double confidence)
		{
			cv::Point3f gaze_direction0_cv(gaze_direction0->Item1, gaze_direction0->Item2, gaze_direction0->Item3);
			cv::Point3f gaze_direction1_cv(gaze_direction1->Item1, gaze_direction1->Item2, gaze_direction1->Item3);

			// Construct an OpenCV matrix from the landmarks
			std::vector<cv::Point2d> landmarks_2D_cv;
			for (int i = 0; i < landmarks_2D->Count; ++i)
			{
				landmarks_2D_cv.push_back(cv::Point2d(landmarks_2D[i]->Item1, landmarks_2D[i]->Item2));
			}

			// Construct an OpenCV matrix from the landmarks
			std::vector<cv::Point3d> landmarks_3D_cv;
			for (int i = 0; i < landmarks_3D->Count; ++i)
			{
				landmarks_3D_cv.push_back(cv::Point3d(landmarks_3D[i]->Item1, landmarks_3D[i]->Item2, landmarks_3D[i]->Item3));
			}

			m_visualizer->SetObservationGaze(gaze_direction0_cv, gaze_direction1_cv, landmarks_2D_cv, landmarks_3D_cv, confidence);
		}

		// Setting the observations
		void SetObservationPose(List<double>^ pose, double confidence)
		{
			cv::Vec6d pose_vec(pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]);
			m_visualizer->SetObservationPose(pose_vec, confidence);
		}

		void SetObservationFaceAlign(OpenCVWrappers::RawImage^ aligned_face_image)
		{
			m_visualizer->SetObservationFaceAlign(aligned_face_image->Mat);
		}

		void SetObservationHOG(OpenCVWrappers::RawImage^ observation_HOG, int num_cols, int num_rows)
		{
			m_visualizer->SetObservationHOG(observation_HOG->Mat, num_cols, num_rows);
		}

		void SetObservationLandmarks(List<System::Tuple<double, double>^>^ landmarks_2D, double confidence, List<bool>^ visibilities)
		{
			// Construct an OpenCV matrix from the landmarks
			cv::Mat_<double> landmarks_2D_mat(landmarks_2D->Count * 2, 1, 0.0);
			for (int i = 0; i < landmarks_2D->Count; ++i)
			{
				landmarks_2D_mat.at<double>(i, 0) = landmarks_2D[i]->Item1;
				landmarks_2D_mat.at<double>(i + landmarks_2D->Count, 0) = landmarks_2D[i]->Item2;
			}

			// Construct an OpenCV matrix from the landmarks
			cv::Mat_<int> visibilities_cv(visibilities->Count, 1, 0);
			for (int i = 0; i < visibilities->Count; ++i)
			{
				if (visibilities[i])
				{
					visibilities_cv.at<int>(i, 0) = 1;
				}
				else
				{
					visibilities_cv.at<int>(i, 0) = 0;
				}
			}

			m_visualizer->SetObservationLandmarks(landmarks_2D_mat, confidence, visibilities_cv);
		}

		void SetObservationLandmarks(List<System::Tuple<double, double>^>^ landmarks_2D, double confidence)
		{
			SetObservationLandmarks(landmarks_2D, confidence, gcnew List<bool>());
		}

		void SetImage(OpenCVWrappers::RawImage^ canvas, float fx, float fy, float cx, float cy)
		{
			m_visualizer->SetImage(canvas->Mat, fx, fy, cx, cy);
		}

		OpenCVWrappers::RawImage^ GetHOGVis()
		{
			OpenCVWrappers::RawImage^ hog_image = gcnew OpenCVWrappers::RawImage(m_visualizer->GetHOGVis());
			return hog_image;
		}

		OpenCVWrappers::RawImage^  GetVisImage()
		{
			OpenCVWrappers::RawImage^ vis_image = gcnew OpenCVWrappers::RawImage(m_visualizer->GetVisImage());
			return vis_image;
		}

		// Finalizer. Definitely called before Garbage Collection,
		// but not automatically called on explicit Dispose().
		// May be called multiple times.
		!Visualizer()
		{
			// Automatically closes capture object before freeing memory.	
			if (m_visualizer != nullptr)
			{
				delete m_visualizer;
			}

		}

		// Destructor. Called on explicit Dispose() only.
		~Visualizer()
		{
			this->!Visualizer();
		}
	};

}
