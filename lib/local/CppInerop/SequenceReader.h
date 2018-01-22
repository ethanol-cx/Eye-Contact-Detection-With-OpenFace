///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis, all rights reserved.
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

#include <opencv2/core/core.hpp>
#include "opencv2/objdetect.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <vector>
#include <set>

#include <OpenCVWrappers.h>
#include <ImageReader.h>

#include "SequenceCapture.h"

#pragma managed

#include <msclr\marshal.h>
#include <msclr\marshal_cppstd.h>

namespace UtilitiesOF {


	public ref class SequenceReader
	{
	private:

		// OpenCV based video capture for reading from files
		Utilities::SequenceCapture* m_sequence_capture;

		OpenCVWrappers::RawImage^ m_rgb_frame;
		OpenCVWrappers::RawImage^ m_gray_frame;

	public:

		// Can provide a directory or a video filename, need to specify which
		SequenceReader(System::String^ filename, bool directory)
		{
			m_sequence_capture = new Utilities::SequenceCapture();

			std::string name_std = msclr::interop::marshal_as<std::string>(filename);

			bool success;

			if(directory)
			{
				success = m_sequence_capture->OpenImageSequence(name_std);
			}
			else
			{
				success = m_sequence_capture->OpenVideoFile(name_std);
			}

			if (!success)
			{
				throw gcnew ReadingFailedException("Failed to open an image sequence");
			}
		}

		// Can provide a webcam id
		SequenceReader(int webcam_id)
		{
			m_sequence_capture = new Utilities::SequenceCapture();
			
			bool success = m_sequence_capture->OpenWebcam(webcam_id);

			if (!success)
			{
				throw gcnew ReadingFailedException("Failed to open an image sequence");
			}
		}

		OpenCVWrappers::RawImage^ GetNextImage()
		{
			cv::Mat next_image = m_sequence_capture->GetNextFrame();

			if (m_rgb_frame == nullptr)
			{
				m_rgb_frame = gcnew OpenCVWrappers::RawImage(next_image.size().width, next_image.size().width, CV_8UC3);
			}

			next_image.copyTo(m_rgb_frame->Mat);

			return m_rgb_frame;
		}

		System::String^ GetName()
		{
			std::string filename = m_sequence_capture->name;
			return gcnew System::String(filename.c_str());
		}

		double GetProgress()
		{
			return m_sequence_capture->GetProgress();
		}

		float GetFx()
		{
			return m_sequence_capture->fx;
		}

		float GetFy()
		{
			return m_sequence_capture->fy;
		}

		float GetCx()
		{
			return m_sequence_capture->cx;
		}

		float GetCy()
		{
			return m_sequence_capture->cy;
		}

		bool IsOpened()
		{
			return m_sequence_capture->IsOpened();
		}

		bool IsWebcam()
		{
			return m_sequence_capture->IsWebcam();
		}

		double GetFPS()
		{
			return m_sequence_capture->fps;
		}

		OpenCVWrappers::RawImage^ GetCurrentFrameGray() {

			cv::Mat next_gray_image = m_sequence_capture->GetGrayFrame();

			if (m_gray_frame == nullptr)
			{
				m_gray_frame = gcnew OpenCVWrappers::RawImage(next_gray_image.size().width, next_gray_image.size().width, CV_8UC3);
			}

			next_gray_image.copyTo(m_gray_frame->Mat);

			return m_gray_frame;
		}

		// Finalizer. Definitely called before Garbage Collection,
		// but not automatically called on explicit Dispose().
		// May be called multiple times.
		!SequenceReader()
		{
			// Automatically closes capture object before freeing memory.	
			if (m_sequence_capture != nullptr)
			{
				delete m_sequence_capture;
			}

			if (m_rgb_frame != nullptr)
			{
				delete m_rgb_frame;
			}
			if (m_gray_frame != nullptr)
			{
				delete m_gray_frame;
			}

		}

		// Destructor. Called on explicit Dispose() only.
		~SequenceReader()
		{
			this->!SequenceReader();
		}
	};

}
