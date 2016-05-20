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

// OpenCVWrappers.h

#pragma once

#pragma managed

#include <msclr\marshal_cppstd.h>

#pragma unmanaged

#include "cv.h"
#include "highgui.h"

#include <opencv2/videoio/videoio.hpp>  // Video write
#include <opencv2/videoio/videoio_c.h>  // Video write

#pragma managed

#pragma make_public(cv::Mat)

using namespace System::Windows;
using namespace System::Windows::Threading;
using namespace System::Windows::Media;
using namespace System::Windows::Media::Imaging;

namespace OpenCVWrappers {

	public ref class RawImage : System::IDisposable
	{

	private:

		cv::Mat* mat;

		static int refCount;


	public:

		static int PixelFormatToType(PixelFormat fmt)
		{
			if (fmt == PixelFormats::Gray8)
				return CV_8UC1;
			else if(fmt == PixelFormats::Bgr24)
				return CV_8UC3;
			else if(fmt == PixelFormats::Bgra32)
				return CV_8UC4;
			else if(fmt == PixelFormats::Gray32Float)
				return CV_32FC1;
			else
				throw gcnew System::Exception("Unsupported pixel format");
		}

		static PixelFormat TypeToPixelFormat(int type)
		{
			switch (type) {
			case CV_8UC1:
				return PixelFormats::Gray8;
			case CV_8UC3:
				return PixelFormats::Bgr24;
			case CV_8UC4:
				return PixelFormats::Bgra32;
			case CV_32FC1:
				return PixelFormats::Gray32Float;
			default:
				throw gcnew System::Exception("Unsupported image type");
			}
		}

		static property int RefCount {
			int get() { return refCount; }
		}

		RawImage() 
		{
			mat = new cv::Mat();
			refCount++;
		}

		RawImage(const cv::Mat& m)
		{
			mat = new cv::Mat(m.clone());
			refCount++;
		}

		RawImage(RawImage^ img)
		{
			mat = new cv::Mat(img->Mat.clone());
			refCount++;
		}

		RawImage(int width, int height, int type)
		{
			mat = new cv::Mat(height, width, type);
			refCount++;
		}

		RawImage(int width, int height, PixelFormat format)
		{
			int type = RawImage::PixelFormatToType(format);
			mat = new cv::Mat(height, width, type);
			refCount++;
		}

		// Finalizer. Definitely called before Garbage Collection,
		// but not automatically called on explicit Dispose().
		// May be called multiple times.
		!RawImage()
		{
			if (mat)
			{
				delete mat;
				mat = NULL;
				refCount--;
			}
		}

		// Destructor. Called on explicit Dispose() only.
		~RawImage()
		{
			this->!RawImage();
		}

		property int Width
		{
			int get()
			{
				return mat->cols;
			}
		}

		property int Height
		{
			int get()
			{
				return mat->rows;
			}
		}

		property int Stride
		{

			int get()
			{
				return mat->step;
			}
		}

		property PixelFormat Format
		{
			PixelFormat get()
			{
				return TypeToPixelFormat(mat->type());
			}
		}

		property cv::Mat& Mat
		{
			cv::Mat& get()
			{
				return *mat;
			}
		}

		property bool IsEmpty
		{
			bool get()
			{
				return !mat || mat->empty();
			}
		}

		bool UpdateWriteableBitmap(WriteableBitmap^ bitmap)
		{
			if (bitmap == nullptr || bitmap->PixelWidth != Width || bitmap->PixelHeight != Height || bitmap->Format != Format)
				return false;
			else {
				if (mat->data == NULL) {
					cv::Mat zeros(bitmap->PixelHeight, bitmap->PixelWidth, PixelFormatToType(bitmap->Format), 0);
					bitmap->WritePixels(Int32Rect(0, 0, Width, Height), System::IntPtr(zeros.data), Stride * Height * (Format.BitsPerPixel / 8), Stride, 0, 0);
				}
				else {
					bitmap->WritePixels(Int32Rect(0, 0, Width, Height), System::IntPtr(mat->data), Stride * Height * (Format.BitsPerPixel / 8), Stride, 0, 0);
				}
				return true;
			}
		}

		WriteableBitmap^ CreateWriteableBitmap()
		{
			return gcnew WriteableBitmap(Width, Height, 72, 72, Format, nullptr);
		}

	};
	
	public ref class VideoWriter
	{
	private:
		// OpenCV based video capture for reading from files
		cv::VideoWriter* vc;

	public:

		VideoWriter(System::String^ location, int width, int height, double fps, bool colour)
		{

			msclr::interop::marshal_context context;
			std::string location_std_string = context.marshal_as<std::string>(location);

			vc = new cv::VideoWriter(location_std_string, CV_FOURCC('D','I','V','X'), fps, cv::Size(width, height), colour);

		}
		
		// Return success
		bool Write(RawImage^ img)
		{
			if(vc != nullptr && vc->isOpened())
			{
				vc->write(img->Mat);
				return true;
			}
			else
			{
				return false;
			}
		}

		// Finalizer. Definitely called before Garbage Collection,
		// but not automatically called on explicit Dispose().
		// May be called multiple times.
		!VideoWriter()
		{
			if(vc != nullptr)
			{
				vc->~VideoWriter();
			}
		}

		// Destructor. Called on explicit Dispose() only.
		~VideoWriter()
		{
			this->!VideoWriter();
		}

	};

}
