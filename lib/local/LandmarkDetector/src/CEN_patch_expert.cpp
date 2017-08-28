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

#include "stdafx.h"

#include "CEN_patch_expert.h"

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

// Local includes
#include "LandmarkDetectorUtils.h"

// OpenBLAS
#include <cblas.h>
#include <f77blas.h>

using namespace LandmarkDetector;

// Copy constructor		
CEN_patch_expert::CEN_patch_expert(const CEN_patch_expert& other) : confidence(other.confidence), width(other.width), height(other.height)
{

	// Copy the layer weights in a deep way
	for (size_t i = 0; i < other.weights.size(); ++i)
	{
		this->weights.push_back(other.weights[i].clone());
		this->biases.push_back(other.biases[i].clone());
		this->activation_function.push_back(other.activation_function[i]);
	}

}

//===========================================================================
void CEN_patch_expert::Read(ifstream &stream)
{

	// Setting up OpenBLAS
	#if _WIN32 || _WIN64
		openblas_set_num_threads(1);
	#endif


	// Sanity check
	int read_type;

	stream.read((char*)&read_type, 4);
	assert(read_type == 6);

	// the number of neurons for this patch
	int num_layers;
	stream.read((char*)&width, 4);
	stream.read((char*)&height, 4);
	stream.read((char*)&num_layers, 4);

	if (num_layers == 0)
	{
		// empty patch due to landmark being invisible at that orientation

		// read an empty int (due to the way things were written out)
		stream.read((char*)&num_layers, 4);
		return;
	}

	activation_function.resize(num_layers);
	weights.resize(num_layers);
	biases.resize(num_layers);

	for (int i = 0; i < num_layers; i++)
	{
		int neuron_type;
		stream.read((char*)&neuron_type, 4);
		activation_function[i] = neuron_type;

		cv::Mat_<double> bias;
		LandmarkDetector::ReadMatBin(stream, bias);

		cv::Mat_<double> weight;
		LandmarkDetector::ReadMatBin(stream, weight);

		weights[i] = weight;
		biases[i] = bias;
	}

	// Read the patch confidence
	stream.read((char*)&confidence, 8);

}

// Contrast normalize the input for response map computation
void contrastNorm(const cv::Mat_<float>& input, cv::Mat_<float>& output)
{

	int num_cols = input.cols;

	int num_rows = input.rows;

	output = input.clone();

	cv::MatConstIterator_<float> p = input.begin();

	// Compute row wise
	for (size_t y = 0; y < num_rows; ++y)
	{
		
		cv::Scalar mean_s = cv::mean(input(cv::Rect(1,y,num_cols-1, 1)));
		double mean = mean_s[0];

		p++;

		float sum_sq = 0;
		for (size_t x = 1; x < num_cols; ++x)
		{
			float curr = *p++;
			sum_sq += (curr - mean) * (curr - mean);
		}

		float norm = sqrt(sum_sq);

		if (norm == 0)
			norm = 1;

		for (size_t x = 1; x < num_cols; ++x)
		{
			output.at<float>(y, x) = (output.at<float>(y, x) - mean) / norm;
		}

	}

}

void im2colBias(const cv::Mat_<float>& input, int width, int height, cv::Mat_<float>& output)
{

	int m = input.rows;
	int n = input.cols;

	// determine how many blocks there will be with a sliding window of width x height in the input
	int yB = m - height + 1;
	int xB = n - width + 1;

	// Allocate the output size
	if(output.rows != xB*yB && output.cols != width * height + 1)
	{
		output = cv::Mat::ones(xB*yB, width * height + 1, CV_32F);
	}

	// Iterate over the blocks
	for (int j = 0; j< xB; j++)
	{
		for (int i = 0; i< yB; i++)
		{
			int rowIdx = i + j*yB;

			for (unsigned int yy = 0; yy < height; ++yy)
				for (unsigned int xx = 0; xx < width; ++xx)
				{
					int colIdx = xx*height + yy;
					output.at<float>(rowIdx, colIdx + 1) = input.at<float>(i + yy, j + xx);
				}
		}
	}
}

//===========================================================================
void CEN_patch_expert::Response(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response)
{

	int response_height = area_of_interest.rows - height + 1;
	int response_width = area_of_interest.cols - width + 1;
	
	cv::Mat_<float> input_col;
	im2colBias(area_of_interest, width, height, input_col);

	// Mean and standard deviation normalization
	contrastNorm(input_col, response);
	cv::Mat_<float> response_blas = response.clone();

	for (size_t layer = 0; layer < activation_function.size(); ++layer)
	{

		// We are performing response = response * weights[layers], but in OpenBLAS as that is significantly quicker than OpenCV
		response_blas = response.clone();

		float* m1 = (float*)response_blas.data;
		float* m2 = (float*)weights[layer].data;
		
		cv::Mat_<float> resp_blas(response_blas.rows, weights[layer].cols, 1.0);
		float* m3 = (float*)resp_blas.data;

		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, weights[layer].cols, response.rows, response.cols, 1, m2, weights[layer].cols, m1, response.cols, 0.0, m3, weights[layer].cols);

		// TODO check speed
		float alpha = 1.0;
		float beta = 0.0;
		sgemm_("N", "N", &weights[layer].cols, &response.rows, &response.cols, &alpha, m2, &weights[layer].cols, m1, &response.cols, &beta, m3, &weights[layer].cols);

		response = resp_blas;

		// TODO bias could be pre-allocated to the window size so that addition could be quicker
		for (size_t y = 0; y < response.rows; ++y)
		{
			response(cv::Rect(0, y, response.cols, 1)) = response(cv::Rect(0, y, response.cols, 1)) + biases[layer];
		}

		// Perform activation		
		if (activation_function[layer] == 0) // Sigmoid
		{			
			for (cv::MatIterator_<float> p = response.begin(); p != response.end(); p++)
			{
				*p = 1.0 / (1.0 + exp(-(*p)));
			}
		}
		else if(activation_function[layer] == 2)// ReLU
		{
			cv::threshold(response, response, 0, 0, cv::THRESH_TOZERO);
		}

	}

	response = response.reshape(1, response_height);
	response = response.t();

}

void im2colBiasSparse(const cv::Mat_<float>& input, int width, int height, cv::Mat_<float>& output)
{

	int m = input.rows;
	int n = input.cols;

	// determine how many blocks there will be with a sliding window of width x height in the input
	int yB = m - height + 1;
	int xB = n - width + 1;

	// As we will be skipping half of the outputs
	int out_size = (yB*xB - 1) / 2;

	// Allocate the output size
	if (output.rows != out_size && output.cols != width * height + 1)
	{
		output = cv::Mat::ones(out_size, width * height + 1, CV_32F);
	}

	// Iterate over the blocks, skipping every second block
	int rowIdx = 0;
	int skipCounter = 0;
	for (int j = 0; j< xB; j++)
	{
		for (int i = 0; i< yB; i++)
		{
			// Skip every second row
			skipCounter++;
			if ((skipCounter + 1) % 2 == 0)
			{
				continue;
			}

			for (unsigned int yy = 0; yy < height; ++yy)
			{
				for (unsigned int xx = 0; xx < width; ++xx)
				{
					int colIdx = xx*height + yy;
					output.at<float>(rowIdx, colIdx + 1) = input.at<float>(i + yy, j + xx);
				}
			}
			rowIdx++;
		}
	}
}

//===========================================================================
//void CEN_patch_expert::ResponseSparse_ocv(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response)
//{
//
//	int response_height = area_of_interest.rows - height + 1;
//	int response_width = area_of_interest.cols - width + 1;
//
//	cv::Mat_<float> input_col;
//	// Extract im2col but in a sparse way
//	im2colBiasSparse(area_of_interest, width, height, input_col);
//
//	// Mean and standard deviation normalization
//	contrastNorm(input_col, response);
//
//	cv::Mat_<float> response_blas = response.clone();
//
//	for (size_t layer = 0; layer < activation_function.size(); ++layer)
//	{
//
//		// We are performing response = response * weights[layers], but in OpenBLAS as that is significantly quicker than OpenCV
//		response_blas = response.clone();
//
//		float* m1 = (float*)response_blas.data;
//		float* m2 = (float*)weights[layer].data;
//
//		cv::Mat_<float> resp_blas(response_blas.rows, weights[layer].cols, 1.0);
//		float* m3 = (float*)resp_blas.data;
//
//		cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, weights[layer].cols, response.rows, response.cols, 1, m2, weights[layer].cols, m1, response.cols, 0.0, m3, weights[layer].cols);
//
//		response = resp_blas;
//
//		// TODO bias could be pre-allocated to the window size so that addition could be quicker
//		for (size_t y = 0; y < response.rows; ++y)
//		{
//			response(cv::Rect(0, y, response.cols, 1)) = response(cv::Rect(0, y, response.cols, 1)) + biases[layer];
//		}
//
//		// Perform activation		
//		if (activation_function[layer] == 0) // Sigmoid
//		{
//			for (cv::MatIterator_<float> p = response.begin(); p != response.end(); p++)
//			{
//				*p = 1.0 / (1.0 + exp(-(*p)));
//			}
//		}
//		else if (activation_function[layer] == 2)// ReLU
//		{
//			cv::threshold(response, response, 0, 0, cv::THRESH_TOZERO);
//		}
//
//	}
//
//	// Restructure the output with interpolation
//	cv::Mat_<float> mapMatrix(response.rows, response_height * response_width, 0.0f);
//
//	// Find a mapping from indices in the computed sparse response and the original full response
//	cv::Mat_<int> value_id_matrix(response_width, response_height, 0);
//
//	int ind = 0;
//	for (int k = 0; k < value_id_matrix.rows * value_id_matrix.cols; ++k)
//	{
//		if (k % 2 != 0)
//		{
//			value_id_matrix.at<int>(k) = ind;
//			ind++;
//		}
//	}
//	value_id_matrix = value_id_matrix.t();
//
//	int skip_counter = 0;
//	for (int x = 0; x < response_width; ++x)
//	{
//		for (int y = 0; y < response_height; ++y)
//		{
//			int mapping_col = x * response_height + y;
//			skip_counter++;
//			if (skip_counter % 2 == 0)
//			{
//				int val_id = value_id_matrix.at<int>(y, x);
//				mapMatrix.at<float>(val_id, mapping_col) = 1;
//				continue;
//			}
//
//			double num_neigh = 0.0;
//			vector<int> val_ids;
//			if (x - 1 >= 0)
//			{
//				num_neigh++;
//				val_ids.push_back(value_id_matrix.at<int>(y, x - 1));
//			}
//			if (y - 1 >= 0)
//			{
//				num_neigh++;
//				val_ids.push_back(value_id_matrix.at<int>(y - 1, x));
//			}
//			if (x + 1 < response_width)
//			{
//				num_neigh++;
//				val_ids.push_back(value_id_matrix.at<int>(y, x + 1));
//			}
//			if (y + 1 < response_height)
//			{
//				num_neigh++;
//				val_ids.push_back(value_id_matrix.at<int>(y + 1, x));
//			}
//
//			for (size_t k = 0; k < val_ids.size(); ++k)
//			{
//				mapMatrix.at<float>(val_ids[k], mapping_col) = 1.0 / num_neigh;
//			}
//		}
//	}
//
//	response = response.t() * mapMatrix;
//	response = response.t();
//	response = response.reshape(1, response_height);
//	response = response.t();
//}

//===========================================================================
void CEN_patch_expert::ResponseSparse(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response)
{

	int response_height = area_of_interest.rows - height + 1;
	int response_width = area_of_interest.cols - width + 1;

	cv::Mat_<float> input_col;
	// Extract im2col but in a sparse way
	im2colBiasSparse(area_of_interest, width, height, input_col);

	// Mean and standard deviation normalization
	contrastNorm(input_col, response);

	cv::Mat_<float> response_blas = response.clone();

	for (size_t layer = 0; layer < activation_function.size(); ++layer)
	{

		// We are performing response = response * weights[layers], but in OpenBLAS as that is significantly quicker than OpenCV
		response_blas = response.clone();

		float* m1 = (float*)response_blas.data;
		float* m2 = (float*)weights[layer].data;

		cv::Mat_<float> resp_blas(response_blas.rows, weights[layer].cols, 1.0);
		float* m3 = (float*)resp_blas.data;


		// TODO check speed
		float alpha = 1.0;
		float beta = 0.0;
		sgemm_("N", "N", &weights[layer].cols, &response.rows, &response.cols, &alpha, m2, &weights[layer].cols, m1, &response.cols, &beta, m3, &weights[layer].cols);

		// The above is a faster version of this, by calling the fortran version directly
		//cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, weights[layer].cols, response.rows, response.cols, 1, m2, weights[layer].cols, m1, response.cols, 0.0, m3, weights[layer].cols);

		response = resp_blas;

		// TODO bias could be pre-allocated to the window size so that addition could be quicker
		for (size_t y = 0; y < response.rows; ++y)
		{
			response(cv::Rect(0, y, response.cols, 1)) = response(cv::Rect(0, y, response.cols, 1)) + biases[layer];
		}

		// Perform activation		
		if (activation_function[layer] == 0) // Sigmoid
		{
			for (cv::MatIterator_<float> p = response.begin(); p != response.end(); p++)
			{
				*p = 1.0 / (1.0 + exp(-(*p)));
			}
		}
		else if (activation_function[layer] == 2)// ReLU
		{
			cv::threshold(response, response, 0, 0, cv::THRESH_TOZERO);
		}

	}
	
	// Restructure the output with interpolation
	cv::Mat_<float> mapMatrix(response.rows, response_height * response_width, 0.0f);

	// Find a mapping from indices in the computed sparse response and the original full response
	cv::Mat_<int> value_id_matrix(response_width, response_height, 0);

	int ind = 0;
	for (int k = 0; k < value_id_matrix.rows * value_id_matrix.cols; ++k)
	{
		if(k % 2 != 0)
		{
			value_id_matrix.at<int>(k) = ind;
			ind++;
		}
	}
	value_id_matrix = value_id_matrix.t();

	int skip_counter = 0;
	for (int x = 0; x < response_width; ++x)
	{
		for (int y = 0; y < response_height; ++y)
		{
			int mapping_col = x * response_height + y;
			skip_counter++;
			if (skip_counter % 2 == 0)
			{
				int val_id = value_id_matrix.at<int>(y, x);
				mapMatrix.at<float>(val_id, mapping_col) = 1;
				continue;
			}

			double num_neigh = 0.0;
			vector<int> val_ids;
			if (x - 1 >= 0)
			{
				num_neigh++;
				val_ids.push_back(value_id_matrix.at<int>(y, x - 1));
			}
			if (y - 1 >= 0)
			{
				num_neigh++;
				val_ids.push_back(value_id_matrix.at<int>(y - 1, x));
			}
			if (x + 1 < response_width)
			{
				num_neigh++;
				val_ids.push_back(value_id_matrix.at<int>(y, x + 1));
			}
			if (y + 1 < response_height)
			{
				num_neigh++;
				val_ids.push_back(value_id_matrix.at<int>(y+1, x));
			}

			for (size_t k = 0; k < val_ids.size(); ++k)
			{
				mapMatrix.at<float>(val_ids[k], mapping_col) = 1.0 / num_neigh;
			}
		}
	}

	response = response.t() * mapMatrix;
	response = response.t();
	response = response.reshape(1, response_height);
	response = response.t();
}
