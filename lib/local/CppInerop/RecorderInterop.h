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

// Camera_Interop.h

#pragma once

#pragma unmanaged

// Include all the unmanaged things we need.

#include "RecorderOpenFace.h"

#pragma managed

namespace Utilities {

	public ref class RecorderOpenFace
	{
	private:

		// OpenCV based video capture for reading from files
		Utilities::RecorderOpenFace* m_recorder;

	public:

		// Can provide a directory, or a list of files
		RecorderOpenFace()
		{
		}


		// Finalizer. Definitely called before Garbage Collection,
		// but not automatically called on explicit Dispose().
		// May be called multiple times.
		!RecorderOpenFace()
		{
			// Automatically closes capture object before freeing memory.	
			if (m_recorder != nullptr)
			{
				delete m_recorder;
			}

		}

		// Destructor. Called on explicit Dispose() only.
		~RecorderOpenFace()
		{
			this->!RecorderOpenFace();
		}
	};

}
