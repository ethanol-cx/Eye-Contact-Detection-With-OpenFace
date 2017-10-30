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
#ifndef __FACE_ANALYSER_INTEROP_h_
#define __FACE_ANALYSER_INTEROP_h_

#pragma once

// Include all the unmanaged things we need.
#pragma managed

#include <msclr\marshal.h>
#include <msclr\marshal_cppstd.h>

#pragma unmanaged

// Allows to overcome boost name clash stuff with C++ CLI
#ifdef __cplusplus_cli
#define generic __identifier(generic)
#endif

#include <opencv2/core/core.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <OpenCVWrappers.h>
#include <Face_utils.h>
#include <FaceAnalyser.h>

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

	// Variables used for recording things
	std::ofstream* hog_output_file;
	std::string* align_output_dir;
	int* num_rows;
	int* num_cols;
	bool* good_frame;

public:

	FaceAnalyserManaged(System::String^ root, bool dynamic, int output_width) 
	{
		string root_std = msclr::interop::marshal_as<std::string>(root);
		FaceAnalysis::FaceAnalyserParameters params(root_std);
		params.setAlignedOutput(output_width);
		face_analyser = new FaceAnalysis::FaceAnalyser(params);

		hog_features = new cv::Mat_<double>();

		aligned_face = new cv::Mat();
		visualisation = new cv::Mat();

		num_rows = new int;
		num_cols = new int;

		good_frame = new bool;
			
		align_output_dir = new string();

		hog_output_file = new std::ofstream();

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

	void StopHOGRecording()
	{
		hog_output_file->close();
	}

	void RecordAlignedFrame(int frame_num)
	{
		char name[100];
					
		// output the frame number
		sprintf(name, "frame_det_%06d.bmp", frame_num);
				
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
	
	void PostProcessOutputFile(System::String^ file)
	{		
		face_analyser->PostprocessOutputFile(msclr::interop::marshal_as<std::string>(file));
	}

	void AddNextFrame(OpenCVWrappers::RawImage^ frame, System::Collections::Generic::List<System::Tuple<double, double>^>^ landmarks, bool success, bool online, bool vis_hog) {
			
		// Construct an OpenCV matric from the landmarks
		cv::Mat_<double> landmarks_mat(landmarks->Count * 2, 1, 0.0);
		for (int i = 0; i < landmarks->Count; ++i)
		{
			landmarks_mat.at<double>(i, 0) = landmarks[i]->Item1;
			landmarks_mat.at<double>(i + landmarks->Count, 0) = landmarks[i]->Item2;
		}

		face_analyser->AddNextFrame(frame->Mat, landmarks_mat, success, 0, online, vis_hog);

		face_analyser->GetLatestHOG(*hog_features, *num_rows, *num_cols);
		
		face_analyser->GetLatestAlignedFace(*aligned_face);
		
		*good_frame = success;

		if(vis_hog)
		{
			*visualisation = face_analyser->GetLatestHOGDescriptorVisualisation();
		}

	}
	
	// Predicting AUs from a single image
	System::Collections::Generic::Dictionary<System::String^, double>^ PredictStaticAUs(OpenCVWrappers::RawImage^ frame, System::Collections::Generic::List<System::Tuple<double, double>^>^ landmarks, bool success, bool vis_hog) {
		
		// Construct an OpenCV matric from the landmarks
		cv::Mat_<double> landmarks_mat(landmarks->Count * 2, 1, 0.0);
		for (int i = 0; i < landmarks->Count; ++i)
		{
			landmarks_mat.at<double>(i, 0) = landmarks[i]->Item1;
			landmarks_mat.at<double>(i + landmarks->Count, 0) = landmarks[i]->Item2;
		}

		face_analyser->AddNextFrame(frame->Mat, landmarks_mat, success, 0, false, vis_hog);

		face_analyser->GetLatestHOG(*hog_features, *num_rows, *num_cols);

		face_analyser->GetLatestAlignedFace(*aligned_face);

		*good_frame = success;

		if (vis_hog)
		{
			*visualisation = face_analyser->GetLatestHOGDescriptorVisualisation();
		}

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
	}

	// Destructor. Called on explicit Dispose() only.
	~FaceAnalyserManaged()
	{
		this->!FaceAnalyserManaged();
	}

};
}

#endif