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

// Camera_Interop.h

#pragma once

#pragma unmanaged

// Include all the unmanaged things we need.

#include <opencv2/core/core.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <set>

#include <OpenCVWrappers.h>

// For camera listings
#include "comet_auto_mf.h"
#include "camera_helper.h"

#pragma managed

#include <msclr\marshal.h>
#include <msclr\marshal_cppstd.h>

namespace CameraInterop {

	public ref class CaptureFailedException : System::Exception 
	{
        public:
        
		CaptureFailedException(System::String^ message): Exception(message){}	
	};
	
	public ref class Capture
	{
	private:

		// OpenCV based video capture for reading from files
		cv::VideoCapture* vc;

		OpenCVWrappers::RawImage^ latestFrame;
		OpenCVWrappers::RawImage^ grayFrame;

		double fps;

		bool is_webcam;
		bool is_image_seq;

		int  frame_num;
		std::vector<std::string>* image_files;

		int vid_length;

	public:

		int width, height;

		Capture(int device, int width, int height)
		{
			assert(device >= 0);

			latestFrame = gcnew OpenCVWrappers::RawImage();

			vc = new cv::VideoCapture(device);
			vc->set(CV_CAP_PROP_FRAME_WIDTH, width);
			vc->set(CV_CAP_PROP_FRAME_HEIGHT, height);

			is_webcam = true;
			is_image_seq = false;

			this->width = width;
			this->height = height;

			vid_length = 0;
			frame_num = 0;

			int set_width = vc->get(CV_CAP_PROP_FRAME_WIDTH);
			int set_height = vc->get(CV_CAP_PROP_FRAME_HEIGHT);

			if(!vc->isOpened())
			{
				throw gcnew CaptureFailedException("Failed to open the webcam");
			}
			if(set_width != width || set_height != height)
			{
				throw gcnew CaptureFailedException("Failed to open the webcam with desired resolution");
			}
		}

		Capture(System::String^ videoFile)
		{
			latestFrame = gcnew OpenCVWrappers::RawImage();

			vc = new cv::VideoCapture(msclr::interop::marshal_as<std::string>(videoFile));
			fps = vc->get(CV_CAP_PROP_FPS);
			is_webcam = false;
			is_image_seq = false;
			this->width = vc->get(CV_CAP_PROP_FRAME_WIDTH);
			this->height = vc->get(CV_CAP_PROP_FRAME_HEIGHT);

			vid_length = vc->get(CV_CAP_PROP_FRAME_COUNT);
			frame_num = 0;

			if(!vc->isOpened())
			{
				throw gcnew CaptureFailedException("Failed to open the video file");
			}
		}

		// An alternative to using video files is using image sequences
		Capture(System::Collections::Generic::List<System::String^>^ image_files)
		{
			
			latestFrame = gcnew OpenCVWrappers::RawImage();

			is_webcam = false;
			is_image_seq = true;
			this->image_files = new std::vector<std::string>();

			for(int i = 0; i < image_files->Count; ++i)
			{
				this->image_files->push_back(msclr::interop::marshal_as<std::string>(image_files[i]));
			}
			vid_length = image_files->Count;
		}

		static System::Collections::Generic::Dictionary<System::String^, System::Collections::Generic::List<System::Tuple<int,int>^>^>^ GetListingFromFile(std::string filename)
		{
			// Check what cameras have been written (using OpenCVs XML packages)
			cv::FileStorage fs_read(filename, cv::FileStorage::READ);

			auto managed_camera_list_initial = gcnew System::Collections::Generic::Dictionary<System::String^, System::Collections::Generic::List<System::Tuple<int,int>^>^>();

			cv::FileNode camera_node_list = fs_read["cameras"];

			// iterate through a sequence using FileNodeIterator
			for(size_t idx = 0; idx < camera_node_list.size(); idx++ )
			{
				std::string camera_name = (std::string)camera_node_list[idx]["name"];
				
				cv::FileNode resolution_list = camera_node_list[idx]["resolutions"];
				auto resolutions = gcnew System::Collections::Generic::List<System::Tuple<int, int>^>();
				for(size_t r_idx = 0; r_idx < resolution_list.size(); r_idx++ )
				{
					int x = (int)resolution_list[r_idx]["x"];
					int y = (int)resolution_list[r_idx]["y"];
					resolutions->Add(gcnew System::Tuple<int,int>(x, y));
				}
				managed_camera_list_initial[gcnew System::String(camera_name.c_str())] = resolutions;
			}
			fs_read.release();
			return managed_camera_list_initial;
		}

		static void WriteCameraListingToFile(System::Collections::Generic::Dictionary<System::String^, System::Collections::Generic::List<System::Tuple<int,int>^>^>^ camera_list, std::string filename)
		{
			cv::FileStorage fs("camera_list.xml", cv::FileStorage::WRITE);

			fs << "cameras" << "[";
			for each( System::String^ name_m in camera_list->Keys )
			{

				std::string name = msclr::interop::marshal_as<std::string>(name_m);

				fs << "{:" << "name" << name;
					fs << "resolutions" << "[";
					auto resolutions = camera_list[name_m];
					for(int j = 0; j < resolutions->Count; j++)
					{

						fs << "{:" << "x" << resolutions[j]->Item1 << "y" << resolutions[j]->Item2;
						fs<< "}";
					}
					fs << "]";
				fs << "}";
			}
			fs << "]";
			fs.release();
		}

		static System::Collections::Generic::List<System::Tuple<System::String^, System::Collections::Generic::List<System::Tuple<int,int>^>^, OpenCVWrappers::RawImage^>^>^ GetCameras(System::String^ root_directory_m)
		{
			std::string root_directory = msclr::interop::marshal_as<std::string>(root_directory_m);
			auto managed_camera_list_initial = GetListingFromFile(root_directory + "camera_list.xml");

			auto managed_camera_list = gcnew System::Collections::Generic::List<System::Tuple<System::String^, System::Collections::Generic::List<System::Tuple<int,int>^>^, OpenCVWrappers::RawImage^>^>();

			// Using DirectShow for capturing from webcams (for MJPG as has issues with other formats)
		    comet::auto_mf auto_mf;

			std::vector<camera> cameras = camera_helper::get_all_cameras();
			
			// A Surface Pro specific hack, it seems to list webcams in a weird way
			for (size_t i = 0; i < cameras.size(); ++i)
			{
				cameras[i].activate();
				std::string name = cameras[i].name(); 
				if(name.compare("Microsoft LifeCam Front") == 0)
				{
					cameras.push_back(cameras[i]);
					cameras.erase(cameras.begin() + i);
				}
			}
			

			for (size_t i = 0; i < cameras.size(); ++i)
			{
				cameras[i].activate();
				std::string name = cameras[i].name(); 
				System::String^ name_managed = gcnew System::String(name.c_str());

				// List camera media types
				auto media_types = cameras[i].media_types();

				System::Collections::Generic::List<System::Tuple<int,int>^>^ resolutions;
				std::set<std::pair<int, int>> res_set;

				// If we have them just use pre-loaded resolutions
				if(managed_camera_list_initial->ContainsKey(name_managed))
				{
					resolutions = managed_camera_list_initial[name_managed];
				}
				else
				{
					resolutions = gcnew System::Collections::Generic::List<System::Tuple<int,int>^>();
					for (size_t m = 0; m < media_types.size(); ++m)
					{
						auto media_type_curr = media_types[m];		
						res_set.insert(std::pair<int, int>(std::pair<int,int>(media_type_curr.resolution().width, media_type_curr.resolution().height)));
					}
				}								
				
				// Grab some sample images and confirm the resolutions
				cv::VideoCapture cap1(i);
				// Go through resolutions if they have not been identified
				if(resolutions->Count == 0)
				{
					for (auto beg = res_set.begin(); beg != res_set.end(); ++beg)
					{
						auto resolution = gcnew System::Tuple<int, int>(beg->first, beg->first);

						cap1.set(CV_CAP_PROP_FRAME_WIDTH, resolution->Item1);
						cap1.set(CV_CAP_PROP_FRAME_HEIGHT, resolution->Item2);

						// Add only valid resolutions as API sometimes provides wrong ones
						int set_width = cap1.get(CV_CAP_PROP_FRAME_WIDTH);
						int set_height = cap1.get(CV_CAP_PROP_FRAME_HEIGHT);

						resolution = gcnew System::Tuple<int, int>(set_width, set_height);
						if(!resolutions->Contains(resolution))
						{
							resolutions->Add(resolution);
						}
					}
					managed_camera_list_initial[name_managed] = resolutions;
				}

				cv::Mat sample_img;
				OpenCVWrappers::RawImage^ sample_img_managed = gcnew OpenCVWrappers::RawImage();

				// Now that the resolutions have been identified, pick a camera and create a thumbnail
				if(resolutions->Count > 0)
				{
					int resolution_ind = resolutions->Count / 2;

					if(resolution_ind >= resolutions->Count)
						resolution_ind = resolutions->Count - 1;

					auto resolution = resolutions[resolution_ind];

					cap1.set(CV_CAP_PROP_FRAME_WIDTH, resolution->Item1);
					cap1.set(CV_CAP_PROP_FRAME_HEIGHT, resolution->Item2);

					for (int k = 0; k < 5; ++k)
						cap1.read(sample_img);

					// Flip horizontally
					cv::flip(sample_img, sample_img, 1);
					

				}
				cap1.~VideoCapture();

				sample_img.copyTo(sample_img_managed->Mat);					

				managed_camera_list->Add(gcnew System::Tuple<System::String^, System::Collections::Generic::List<System::Tuple<int,int>^>^, OpenCVWrappers::RawImage^>(gcnew System::String(name.c_str()), resolutions, sample_img_managed));
			}

			WriteCameraListingToFile(managed_camera_list_initial, root_directory + "camera_list.xml");

			return managed_camera_list;
		}

		OpenCVWrappers::RawImage^ GetNextFrame(bool mirror)
		{
			frame_num++;

			if(vc != nullptr)
			{
				
				bool success = vc->read(latestFrame->Mat);

				if (!success)
				{
					// Indicate lack of success by returning an empty image
					cv::Mat empty_mat = cv::Mat();
					empty_mat.copyTo(latestFrame->Mat);
					return latestFrame;
				}
			}
			else if(is_image_seq)
			{
				if(image_files->empty())
				{
					// Indicate lack of success by returning an empty image
					cv::Mat empty_mat = cv::Mat();
					empty_mat.copyTo(latestFrame->Mat);
					return latestFrame;
				}

				cv::Mat img = cv::imread(image_files->at(0), -1);
				img.copyTo(latestFrame->Mat);
				// Remove the first frame
				image_files->erase(image_files->begin(), image_files->begin() + 1);
			}
			
			if (grayFrame == nullptr) {
				if (latestFrame->Width > 0) {
					grayFrame = gcnew OpenCVWrappers::RawImage(latestFrame->Width, latestFrame->Height, CV_8UC1);
				}
			}

			if(mirror)
			{
				flip(latestFrame->Mat, latestFrame->Mat, 1);
			}


			if (grayFrame != nullptr) {
				cvtColor(latestFrame->Mat, grayFrame->Mat, CV_BGR2GRAY);
			}

			return latestFrame;
		}

		double GetProgress()
		{
			if(vc != nullptr && is_webcam)
			{
				return - 1.0;
			}
			else
			{
				return (double)frame_num / (double)vid_length;
			}
		}

		bool isOpened()
		{
			if(vc != nullptr)
				return vc->isOpened();
			else
			{
				if(is_image_seq && image_files->size() > 0)
					return true;
				else
					return false;
			}
		}

		OpenCVWrappers::RawImage^ GetCurrentFrameGray() {
			return grayFrame;
		}

		double GetFPS() {
			return fps;
		}
		
		// Finalizer. Definitely called before Garbage Collection,
		// but not automatically called on explicit Dispose().
		// May be called multiple times.
		!Capture()
		{
			// Automatically closes capture object before freeing memory.	
			if(vc != nullptr)
			{
				vc->~VideoCapture();
			}
			if(image_files != nullptr)
				delete image_files;
		}

		// Destructor. Called on explicit Dispose() only.
		~Capture()
		{
			this->!Capture();
		}
	};

}
