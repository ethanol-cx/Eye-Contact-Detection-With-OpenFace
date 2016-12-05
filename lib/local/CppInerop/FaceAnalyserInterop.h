///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2016, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
//
// THIS SOFTWARE IS PROVIDED “AS IS” FOR ACADEMIC USE ONLY AND ANY EXPRESS
// OR IMPLIED WARRANTIES WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY.
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
// HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
// STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// Notwithstanding the license granted herein, Licensee acknowledges that certain components
// of the Software may be covered by so-called “open source” software licenses (“Open Source
// Components”), which means any software licenses approved as open source licenses by the
// Open Source Initiative or any substantially similar licenses, including without limitation any
// license that, as a condition of distribution of the software licensed under such license,
// requires that the distributor make the software available in source code format. Licensor shall
// provide a list of Open Source Components for a particular version of the Software upon
// Licensee’s request. Licensee will comply with the applicable terms of such licenses and to
// the extent required by the licenses covering Open Source Components, the terms of such
// licenses will apply in lieu of the terms of this Agreement. To the extent the terms of the
// licenses applicable to Open Source Components prohibit any of the restrictions in this
// License Agreement with respect to such Open Source Component, such restrictions will not
// apply to such Open Source Component. To the extent the terms of the licenses applicable to
// Open Source Components require Licensor to make an offer to provide source code or
// related information in connection with the Software, such offer is hereby made. Any request
// for source code or related information should be directed to cl-face-tracker-distribution@lists.cam.ac.uk
// Licensee acknowledges receipt of notices for the Open Source Components for the initial
// delivery of the Software.

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
#ifndef __FACE_ANALYSER_INTEROP_h_
#define __FACE_ANALYSER_INTEROP_h_

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

#include <OpenCVWrappers.h>
#include <LandmarkDetectorInterop.h>
#include <Face_utils.h>
#include <FaceAnalyser.h>
#include <GazeEstimation.h>

// Allows to overcome boost name clash stuff with C++ CLI
#ifdef __cplusplus_cli
#define generic __identifier(generic)
#endif

// Boost stuff
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

#ifdef __cplusplus_cli
#undef generic
#endif

#pragma managed

namespace FaceAnalyser_Interop {

public ref class FaceAnalyserManaged
{

private:

	FaceAnalysis::FaceAnalyser* face_analyser;

	// The actual descriptors (for visualisation and output)
	cv::Mat_<double>* hog_features;
	cv::Mat* aligned_face;
	cv::Mat* visualisation;
	cv::Mat* tracked_face;

	// Variables used for recording things
	std::ofstream* hog_output_file;
	std::string* align_output_dir;
	int* num_rows;
	int* num_cols;
	bool* good_frame;
	cv::VideoWriter* tracked_vid_writer;

	// Variable storing gaze for recording

	// Absolute gaze direction
	cv::Point3f* gazeDirection0;
	cv::Point3f* gazeDirection1;
	cv::Vec2d* gazeAngle;

	cv::Point3f* pupil_left;
	cv::Point3f* pupil_right;

public:

	FaceAnalyserManaged(System::String^ root, bool dynamic, int output_width) 
	{
			
		vector<cv::Vec3d> orientation_bins;
		orientation_bins.push_back(cv::Vec3d(0,0,0));

		int width = output_width;
		int height = output_width;
		
		double scale = width * (0.7 / 112.0);

		string root_std = msclr::interop::marshal_as<std::string>(root);
		
		// TODO diff paths and locations for the demo mode
		boost::filesystem::path tri_loc = boost::filesystem::path(root_std) / "model" / "tris_68_full.txt";
		boost::filesystem::path au_loc;
		if(dynamic)
		{
			au_loc = boost::filesystem::path(root_std) / "AU_predictors" / "AU_all_best.txt";
		}
		else
		{
			au_loc = boost::filesystem::path(root_std) / "AU_predictors" / "AU_all_static.txt";
		}

		face_analyser = new FaceAnalysis::FaceAnalyser(orientation_bins, scale, width, height, au_loc.string(), tri_loc.string());

		hog_features = new cv::Mat_<double>();

		aligned_face = new cv::Mat();
		visualisation = new cv::Mat();
		tracked_face = new cv::Mat();

		num_rows = new int;
		num_cols = new int;

		good_frame = new bool;
			
		align_output_dir = new string();

		hog_output_file = new std::ofstream();

		gazeDirection0 = new cv::Point3f();
		gazeDirection1 = new cv::Point3f();
		gazeAngle = new cv::Vec2d();

		pupil_left = new cv::Point3f();
		pupil_right = new cv::Point3f();
	}

	void SetupAlignedImageRecording(System::String^ directory)
	{
		*align_output_dir = msclr::interop::marshal_as<std::string>(directory);
	}

	void SetupHOGRecording(System::String^ file)
	{
		// Create the file for recording			
		hog_output_file->open(msclr::interop::marshal_as<std::string>(file), ios_base::out | ios_base::binary);
	}

	void SetupTrackingRecording(System::String^ file, int width, int height, double fps)
	{
		tracked_vid_writer = new cv::VideoWriter(msclr::interop::marshal_as<std::string>(file), CV_FOURCC('D', 'I', 'V', 'X'), fps, cv::Size(width, height));
	}

	void StopHOGRecording()
	{
		hog_output_file->close();
	}

	void StopTrackingRecording()
	{
		tracked_vid_writer->release();
	}

	void RecordAlignedFrame(int frame_num)
	{
		char name[100];
					
		// output the frame number
		sprintf(name, "frame_det_%06d.png", frame_num);
				
		string out_file = (boost::filesystem::path(*align_output_dir) / boost::filesystem::path(name)).string();
		imwrite(out_file, *aligned_face);
	}

	void RecordHOGFrame()
	{
		// Using FHOGs, hence 31 channels
		int num_channels = 31;

		hog_output_file->write((char*)(num_cols), 4);
		hog_output_file->write((char*)(num_rows), 4);
		hog_output_file->write((char*)(&num_channels), 4);

		// Not the best way to store a bool, but will be much easier to read it
		float good_frame_float;
		if(good_frame)
			good_frame_float = 1;
		else
			good_frame_float = -1;

		hog_output_file->write((char*)(&good_frame_float), 4);

		cv::MatConstIterator_<double> descriptor_it = hog_features->begin();

		for(int y = 0; y < *num_cols; ++y)
		{
			for(int x = 0; x < *num_rows; ++x)
			{
				for(unsigned int o = 0; o < 31; ++o)
				{

					float hog_data = (float)(*descriptor_it++);
					hog_output_file->write((char*)&hog_data, 4);
				}
			}
		}
		
	}

	void RecordTrackedFace()
	{
		tracked_vid_writer->write(*tracked_face);
	}

	void AddNextFrame(OpenCVWrappers::RawImage^ frame, CppInterop::LandmarkDetector::CLNF^ clnf, double fx, double fy, double cx, double cy, bool online, bool vis_hog, bool vis_tracked) {
			
		face_analyser->AddNextFrame(frame->Mat, *clnf->getCLNF(), 0, online, vis_hog);

		face_analyser->GetLatestHOG(*hog_features, *num_rows, *num_cols);
		
		face_analyser->GetLatestAlignedFace(*aligned_face);
		
		*good_frame = clnf->clnf->detection_success;

		if(vis_hog)
		{
			*visualisation = face_analyser->GetLatestHOGDescriptorVisualisation();
		}

		if(vis_tracked)
		{
			if(frame->Mat.cols != tracked_face->cols && frame->Mat.rows != tracked_face->rows)
			{
				*tracked_face = frame->Mat.clone();
			}
			else
			{
				frame->Mat.clone().copyTo(*tracked_face);
			}

			if(clnf->clnf->detection_success)
			{
				::LandmarkDetector::Draw(*tracked_face, *clnf->clnf);
			}
			tracked_face->deallocate();
		}

		// After the AUs have been detected do some gaze estimation as well
		FaceAnalysis::EstimateGaze(*clnf->getCLNF(), *gazeDirection0, fx, fy, cx, cy, true);
		FaceAnalysis::EstimateGaze(*clnf->getCLNF(), *gazeDirection1, fx, fy, cx, cy, false);

		// Estimate the gaze angle WRT to head pose here
		System::Collections::Generic::List<double>^ pose_list = gcnew System::Collections::Generic::List<double>();
		clnf->GetPose(pose_list, fx, fy, cx, cy);
		cv::Vec6d pose(pose_list[0], pose_list[1], pose_list[2], pose_list[3], pose_list[4], pose_list[5]);

		cv::Vec2d gaze_angle = FaceAnalysis::GetGazeAngle(*gazeDirection0, *gazeDirection1, pose);

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
		cv::Point3f pupil_left_h = FaceAnalysis::GetPupilPosition(eyeLdmks3d_left);
		pupil_left->x = pupil_left_h.x; pupil_left->y = pupil_left_h.y; pupil_left->z = pupil_left_h.z;

		cv::Mat_<double> eyeLdmks3d_right = clnf->getCLNF()->hierarchical_models[part_right].GetShape(fx, fy, cx, cy);
		cv::Point3f pupil_right_h = FaceAnalysis::GetPupilPosition(eyeLdmks3d_right);
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
	System::Collections::Generic::List<System::Tuple<System::Windows::Point, System::Windows::Point>^>^ CalculateGazeLines(double scale, float fx, float fy, float cx, float cy)
	{
		
		cv::Mat_<double> cameraMat = (cv::Mat_<double>(3, 3) << fx, 0, cx, 0, fy, cy, 0, 0, 0);

		vector<cv::Point3f> points_left;
		points_left.push_back(cv::Point3f(*pupil_left));
		points_left.push_back(cv::Point3f(*pupil_left + *gazeDirection0 * 40.0 * scale));

		vector<cv::Point3f> points_right;
		points_right.push_back(cv::Point3f(*pupil_right));
		points_right.push_back(cv::Point3f(*pupil_right + *gazeDirection1 * 40.0 * scale));

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


	System::Collections::Generic::List<System::String^>^ GetClassActionUnitsNames()
	{
		auto names = face_analyser->GetAUClassNames();

		auto names_ret = gcnew System::Collections::Generic::List<System::String^>();

		for(std::string name : names)
		{
			names_ret->Add(gcnew System::String(name.c_str()));
		}

		return names_ret;

	}

	System::Collections::Generic::List<System::String^>^ GetRegActionUnitsNames()
	{
		auto names = face_analyser->GetAURegNames();

		auto names_ret = gcnew System::Collections::Generic::List<System::String^>();

		for(std::string name : names)
		{
			names_ret->Add(gcnew System::String(name.c_str()));
		}

		return names_ret;

	}

	System::Collections::Generic::Dictionary<System::String^, double>^ GetCurrentAUsClass()
	{
		auto classes = face_analyser->GetCurrentAUsClass();
		auto au_classes = gcnew System::Collections::Generic::Dictionary<System::String^, double>();

		for(auto p: classes)
		{
			au_classes->Add(gcnew System::String(p.first.c_str()), p.second);
		}
		return au_classes;
	}

	System::Collections::Generic::Dictionary<System::String^, double>^ GetCurrentAUsReg()
	{
		auto preds = face_analyser->GetCurrentAUsReg();
		auto au_preds = gcnew System::Collections::Generic::Dictionary<System::String^, double>();

		for(auto p: preds)
		{
			au_preds->Add(gcnew System::String(p.first.c_str()), p.second);
		}
		return au_preds;
	}

	OpenCVWrappers::RawImage^ GetLatestAlignedFace() {
		OpenCVWrappers::RawImage^ face_aligned_image = gcnew OpenCVWrappers::RawImage(*aligned_face);
		return face_aligned_image;
	}

	OpenCVWrappers::RawImage^ GetLatestHOGDescriptorVisualisation() {
		OpenCVWrappers::RawImage^ HOG_vis_image = gcnew OpenCVWrappers::RawImage(*visualisation);
		return HOG_vis_image;
	}

	void Reset()
	{
		face_analyser->Reset();
	}

	// Finalizer. Definitely called before Garbage Collection,
	// but not automatically called on explicit Dispose().
	// May be called multiple times.
	!FaceAnalyserManaged()
	{
		delete hog_features;
		delete aligned_face;
		delete visualisation;
		delete num_cols;
		delete num_rows;
		delete hog_output_file;
		delete good_frame;
		delete align_output_dir;
		delete face_analyser;
		delete tracked_face;

		delete gazeDirection0;
		delete gazeDirection1;
		delete gazeAngle;

		delete pupil_left;
		delete pupil_right;

		if(tracked_vid_writer != 0)
		{
			delete tracked_vid_writer;
		}
	}

	// Destructor. Called on explicit Dispose() only.
	~FaceAnalyserManaged()
	{
		this->!FaceAnalyserManaged();
	}

};
}

#endif