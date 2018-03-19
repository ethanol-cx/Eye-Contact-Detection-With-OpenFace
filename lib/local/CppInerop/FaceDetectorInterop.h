///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis,
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

#ifndef __LANDMARK_DETECTOR_UTILS_INTEROP_h_
#define __LANDMARK_DETECTOR_UTILS_INTEROP_h_

#pragma once

// Include all the unmanaged things we need.

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

#include <LandmarkCoreIncludes.h>

#ifdef __cplusplus_cli
#undef generic
#endif

using namespace System::Collections::Generic;

#pragma managed
#include <msclr\marshal.h>
#include <msclr\marshal_cppstd.h>

namespace FaceDetectorInterop {

	public ref class FaceDetector
	{

	private:
		// Where the face detectors are stored
		dlib::frontal_face_detector* face_detector_hog;
		LandmarkDetector::FaceDetectorMTCNN* face_detector_mtcnn;
		cv::CascadeClassifier* face_detector_haar;

	public:

		// The constructor initializes the dlib face detector
		FaceDetector(System::String^ haar_location, System::String^ mtcnn_location) 
		{
			// Initialize all of the detectors (TODO should be done on need only basis)
			face_detector_hog = new dlib::frontal_face_detector(dlib::get_frontal_face_detector());
			face_detector_mtcnn = new LandmarkDetector::FaceDetectorMTCNN(msclr::interop::marshal_as<std::string>(mtcnn_location));
			face_detector_haar = new cv::CascadeClassifier(msclr::interop::marshal_as<std::string>(haar_location));
		}

		// Face detection using HOG-SVM classifier
		void DetectFacesHOG(List<System::Windows::Rect>^ o_regions, OpenCVWrappers::RawImage^ intensity, List<float>^ o_confidences)
		{
			std::vector<cv::Rect_<float> > regions_ocv;
			std::vector<float> confidences_std;

			::LandmarkDetector::DetectFacesHOG(regions_ocv, intensity->Mat, *face_detector_hog, confidences_std);

			o_regions->Clear();
			o_confidences->Clear();

			for(size_t i = 0; i < regions_ocv.size(); ++i)
			{
				o_regions->Add(System::Windows::Rect(regions_ocv[i].x, regions_ocv[i].y, regions_ocv[i].width, regions_ocv[i].height));
				o_confidences->Add(confidences_std[i]);
			}
		}

		// Face detection using MTCNN face detector
		void DetectFacesMTCNN(List<System::Windows::Rect>^ o_regions, OpenCVWrappers::RawImage^ rgb_image, List<float>^ o_confidences)
		{
			std::vector<cv::Rect_<float> > regions_ocv;
			std::vector<float> confidences_std;

			::LandmarkDetector::DetectFacesMTCNN(regions_ocv, rgb_image->Mat, *face_detector_mtcnn, confidences_std);

			o_regions->Clear();
			o_confidences->Clear();

			for (size_t i = 0; i < regions_ocv.size(); ++i)
			{
				o_regions->Add(System::Windows::Rect(regions_ocv[i].x, regions_ocv[i].y, regions_ocv[i].width, regions_ocv[i].height));
				o_confidences->Add(confidences_std[i]);
			}
		}

		// Finalizer. Definitely called before Garbage Collection,
		// but not automatically called on explicit Dispose().
		// May be called multiple times.
		!FaceDetector()
		{
			// TODO need only basis (might be null)
			delete face_detector_hog;
			delete face_detector_mtcnn;
			delete face_detector_haar;
		}

		// Destructor. Called on explicit Dispose() only.
		~FaceDetector()
		{
			this->!FaceDetector();
		}

	};

}

#endif