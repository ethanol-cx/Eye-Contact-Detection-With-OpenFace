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
//
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

#include "CCNF_patch_expert.h"

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

// Local includes
#include "LandmarkDetectorUtils.h"

// OpenBLAS
#include <cblas.h>
#include <f77blas.h>

using namespace LandmarkDetector;

// Copy constructors of neuron and patch expert
CCNF_neuron::CCNF_neuron(const CCNF_neuron& other) : weights(other.weights.clone())
{
	this->neuron_type = other.neuron_type;
	this->norm_weights = other.norm_weights;
	this->bias = other.bias;
	this->alpha = other.alpha;

	for (std::map<int, cv::Mat_<double> >::const_iterator it = other.weights_dfts.begin(); it != other.weights_dfts.end(); it++)
	{
		// Make sure the matrix is copied.
		this->weights_dfts.insert(std::pair<int, cv::Mat>(it->first, it->second.clone()));
	}
}

// Copy constructor		
CCNF_patch_expert::CCNF_patch_expert(const CCNF_patch_expert& other) : neurons(other.neurons), window_sizes(other.window_sizes), betas(other.betas)
{
	this->width = other.width;
	this->height = other.height;
	this->patch_confidence = other.patch_confidence;
	
	this->weight_matrix = other.weight_matrix.clone();

	// Copy the Sigmas in a deep way
	for (std::vector<cv::Mat_<float> >::const_iterator it = other.Sigmas.begin(); it != other.Sigmas.end(); it++)
	{
		// Make sure the matrix is copied.
		this->Sigmas.push_back(it->clone());
	}

}

// Compute sigmas for all landmarks for a particular view and window size
void CCNF_patch_expert::ComputeSigmas(std::vector<cv::Mat_<float> > sigma_components, int window_size)
{
	for(size_t i=0; i < window_sizes.size(); ++i)
	{
		if( window_sizes[i] == window_size)
			return;
	}
	// Each of the landmarks will have the same connections, hence constant number of sigma components
	int n_betas = sigma_components.size();

	// calculate the sigmas based on alphas and betas
	float sum_alphas = 0;

	int n_alphas = this->neurons.size();

	// sum the alphas first
	for(int a = 0; a < n_alphas; ++a)
	{
		sum_alphas = sum_alphas + this->neurons[a].alpha;
	}

	cv::Mat_<float> q1 = sum_alphas * cv::Mat_<float>::eye(window_size*window_size, window_size*window_size);

	cv::Mat_<float> q2 = cv::Mat_<float>::zeros(window_size*window_size, window_size*window_size);
	for (int b=0; b < n_betas; ++b)
	{			
		q2 = q2 + ((float)this->betas[b]) * sigma_components[b];
	}

	cv::Mat_<float> SigmaInv = 2 * (q1 + q2);
	
	cv::Mat Sigma_f;
	cv::invert(SigmaInv, Sigma_f, cv::DECOMP_CHOLESKY);

	window_sizes.push_back(window_size);
	Sigmas.push_back(Sigma_f);

}

//===========================================================================
void CCNF_neuron::Read(ifstream &stream)
{
	// Sanity check
	int read_type;
	stream.read ((char*)&read_type, 4);
	assert(read_type == 2);

	stream.read ((char*)&neuron_type, 4);
	stream.read ((char*)&norm_weights, 8);
	stream.read ((char*)&bias, 8);
	stream.read ((char*)&alpha, 8);
	
	LandmarkDetector::ReadMatBin(stream, weights); 

}

void im2col(const cv::Mat_<float>& input, int width, int height, cv::Mat_<float>& output)
{

	const int m = input.rows;
	const int n = input.cols;

	// determine how many blocks there will be with a sliding window of width x height in the input
	const int yB = m - height + 1;
	const int xB = n - width + 1;

	// Allocate the output size
	if (output.rows != xB*yB && output.cols != width * height)
	{
		output = cv::Mat::zeros(xB*yB, width * height, CV_32F);
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
					output.at<float>(rowIdx, colIdx) = input.at<float>(i + yy, j + xx);
					// TODO could compute the mean here instead of contrast norm
				}
		}
	}
}

// TODO this was optimized more in CEN that can be re-used
// Contrast normalize the input for response map computation (TODO if it works move to utilities)
void contrastNormCCNF(const cv::Mat_<float>& input, cv::Mat_<float>& output)
{

	const int num_cols = input.cols;

	const int num_rows = input.rows;

	output = input.clone();

	cv::MatConstIterator_<float> p = input.begin();

	cv::MatIterator_<float> o = output.begin();

	// Compute row wise
	for (size_t y = 0; y < num_rows; ++y)
	{
		// TODO means could be computed externally
		cv::Scalar mean_s = cv::mean(input(cv::Rect(0, y, num_cols, 1)));
		float mean = (float)mean_s[0];

		float sum_sq = 0;
		// TODO Some of this could be pre-computed? e.g. mean*mean and curr*curr in im2col
		for (size_t x = 0; x < num_cols; ++x)
		{
			float curr = *p++;
			sum_sq += (curr - mean) * (curr - mean);
		}

		float norm = sqrt(sum_sq);

		if (norm == 0)
			norm = 1;
		
		// Faster to multiply
		norm = 1.0 / norm;

		for (size_t x = 0; x < num_cols; ++x)
		{
			//output.at<float>(y, x) = (output.at<float>(y, x) - mean) / norm;
			*o++ = (*o - mean) * norm;
		}

	}

}

// Perform im2col, while at the same time doing contrast normalization and adding a bias term (also skip every other region)
void im2colContrastNorm(const cv::Mat_<float>& input, const int width, const int height, cv::Mat_<float>& output)
{
	const int m = input.rows;
	const int n = input.cols;

	// determine how many blocks there will be with a sliding window of width x height in the input
	const int yB = m - height + 1;
	const int xB = n - width + 1;

	// Allocate the output size
	if (output.rows != xB*yB && output.cols != width * height)
	{
		output = cv::Mat::zeros(xB*yB, width * height, CV_32F);
	}

	// Iterate over the blocks,
	int rowIdx = 0;
	for (int j = 0; j< xB; j++)
	{
		for (int i = 0; i< yB; i++)
		{

			float* Mo = output.ptr<float>(rowIdx);

			float sum = 0;

			for (unsigned int yy = 0; yy < height; ++yy)
			{
				const float* Mi = input.ptr<float>(i + yy);
				for (unsigned int xx = 0; xx < width; ++xx)
				{
					int colIdx = xx*height + yy;
					float in = Mi[j + xx];
					sum += in;

					Mo[colIdx] = in;
				}
			}

			// Working out the mean
			float mean = sum / (float)(width * height);

			float sum_sq = 0;

			// Working out the sum squared and subtracting the mean
			for (size_t x = 0; x < width*height; ++x)
			{
				float in = Mo[x] - mean;
				Mo[x] = in;
				sum_sq += in * in;
			}

			float norm = sqrt(sum_sq);

			// Avoiding division by 0
			if (norm == 0)
			{
				norm = 1;
			}

			// Flip multiplication to division for speed
			norm = 1.0 / norm;

			for (size_t x = 0; x < width*height; ++x)
			{
				Mo[x] *= norm;
			}

			rowIdx++;
		}
	}
}

//===========================================================================
void CCNF_neuron::Response(const cv::Mat_<float> &im, cv::Mat_<double> &im_dft, cv::Mat &integral_img, cv::Mat &integral_img_sq, cv::Mat_<float> &resp)
{

	int h = im.rows - weights.rows + 1;
	int w = im.cols - weights.cols + 1;
	
	// the patch area on which we will calculate reponses
	cv::Mat_<float> I;

	if(neuron_type == 3)
	{
		// Perform normalisation across whole patch (ignoring the invalid values indicated by <= 0

		cv::Scalar mean;
		cv::Scalar std;
		
		// ignore missing values
		cv::Mat_<uchar> mask = im > 0;
		cv::meanStdDev(im, mean, std, mask);

		// if all values the same don't divide by 0
		if(std[0] != 0)
		{
			I = (im - mean[0]) / std[0];
		}
		else
		{
			I = (im - mean[0]);
		}

		I.setTo(0, mask == 0);
	}
	else
	{
		if(neuron_type == 0)
		{
			I = im;
		}
		else
		{
			printf("ERROR(%s,%d): Unsupported patch type %d!\n", __FILE__,__LINE__,neuron_type);
			abort();
		}
	}
  
	if(resp.empty())
	{		
		resp.create(h, w);
	}

	// The response from neuron before activation
	if(neuron_type == 3)
	{
		// In case of depth we use per area, rather than per patch normalisation
		matchTemplate_m(I, im_dft, integral_img, integral_img_sq, weights, weights_dfts, resp, CV_TM_CCOEFF); // the linear multiplication, efficient calc of response
	}
	else
	{
		matchTemplate_m(I, im_dft, integral_img, integral_img_sq, weights, weights_dfts, resp, CV_TM_CCOEFF_NORMED); // the linear multiplication, efficient calc of response
	}
	
	cv::MatIterator_<float> p = resp.begin();

	cv::MatIterator_<float> q1 = resp.begin(); // respone for each pixel
	cv::MatIterator_<float> q2 = resp.end();

	// the logistic function (sigmoid) applied to the response
	while(q1 != q2)
	{
		*p++ = (2 * alpha) * 1.0 /(1.0 + exp( -(*q1++ * norm_weights + bias )));
	}

}



// TODO building up from individual responses (although in future would want to move this up rather than per individual neuron), remove
void CCNF_neuron::ResponseOB(const cv::Mat_<float> &area_of_interest, cv::Mat_<float>& normalized_input, cv::Mat_<float> &resp)
{

	int h = area_of_interest.rows - weights.rows + 1;
	int w = area_of_interest.cols - weights.cols + 1;

	if (neuron_type != 0)
	{
		printf("ERROR(%s,%d): Unsupported patch type %d!\n", __FILE__, __LINE__, neuron_type);
		abort();
	}

	//if (resp.empty())
	//{
	//	resp.create(h, w);
	//}
	
	// Perform im2col and contrast normalization
	if(normalized_input.empty())
	{
		// TODO cleanup
		//im2col(area_of_interest, weights.cols, weights.rows, input_col);

		// Mean and standard deviation normalization
		//contrastNormCCNF(input_col, resp);

		cv::Mat_<float> tmp_out;
		im2colContrastNorm(area_of_interest, weights.cols, weights.rows, normalized_input);
		
		normalized_input = normalized_input.t();
	}

	// Flatten the weights (TODO could pull this out or do it during model reading)
	cv::Mat_<float> w_tmp = weights.t();
	cv::Mat_<float> weights_flat = w_tmp.reshape(1, weights.rows * weights.cols);
	weights_flat = weights_flat.t();

	// Perform computation (TODO OpenBlas it)
	resp = weights_flat * normalized_input;
	resp = resp.reshape(1, h);

	cv::MatIterator_<float> p = resp.begin();

	cv::MatIterator_<float> q1 = resp.begin(); // respone for each pixel
	cv::MatIterator_<float> q2 = resp.end();

	// the logistic function (sigmoid) applied to the response
	while (q1 != q2)
	{
		*p++ = (2 * alpha) * 1.0 / (1.0 + exp(-(*q1++ * norm_weights + bias)));
	}
	resp = resp.t();
}

//===========================================================================
void CCNF_patch_expert::Read(ifstream &stream, std::vector<int> window_sizes, std::vector<std::vector<cv::Mat_<float> > > sigma_components)
{

	// Sanity check
	int read_type;

	stream.read ((char*)&read_type, 4);
	assert(read_type == 5);

	// the number of neurons for this patch
	int num_neurons;
	stream.read ((char*)&width, 4);
	stream.read ((char*)&height, 4);
	stream.read ((char*)&num_neurons, 4);

	if(num_neurons == 0)
	{
		// empty patch due to landmark being invisible at that orientation
	
		// read an empty int (due to the way things were written out)
		stream.read ((char*)&num_neurons, 4);
		return;
	}

	neurons.resize(num_neurons);
	for(int i = 0; i < num_neurons; i++)
		neurons[i].Read(stream);
		
	// Combine the neuron weights to one weight matrix for more efficient computation
	weight_matrix = cv::Mat_<float>(neurons.size(), neurons[0].weights.rows * neurons[0].weights.cols);
	for (size_t i = 0; i < neurons.size(); i++)
	{
		cv::Mat_<float> w_tmp = neurons[i].weights.t();
		cv::Mat_<float> weights_flat = w_tmp.reshape(1, neurons[i].weights.rows * neurons[i].weights.cols);
		weights_flat = weights_flat.t();
		weights_flat.copyTo(weight_matrix(cv::Rect(0, i, neurons[i].weights.rows * neurons[i].weights.cols, 1)));
	}

	int n_sigmas = window_sizes.size();

	int n_betas = 0;

	if(n_sigmas > 0)
	{
		n_betas = sigma_components[0].size();

		betas.resize(n_betas);

		for (int i=0; i < n_betas;  ++i)
		{
			stream.read ((char*)&betas[i], 8);
		}
	}	

	// Read the patch confidence
	stream.read ((char*)&patch_confidence, 8);

}

//===========================================================================
void CCNF_patch_expert::Response(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response)
{
	
	int response_height = area_of_interest.rows - height + 1;
	int response_width = area_of_interest.cols - width + 1;

	if(response.rows != response_height || response.cols != response_width)
	{
		response.create(response_height, response_width);
	}
		
	response.setTo(0);
	
	// the placeholder for the DFT of the image, the integral image, and squared integral image so they don't get recalculated for every response
	cv::Mat_<double> area_of_interest_dft;
	cv::Mat integral_image, integral_image_sq;
	
	cv::Mat_<float> neuron_response;

	// responses from the neural layers
	for(size_t i = 0; i < neurons.size(); i++)
	{		
		// Do not bother with neuron response if the alpha is tiny and will not contribute much to overall result
		if(neurons[i].alpha > 1e-4)
		{

			neurons[i].Response(area_of_interest, area_of_interest_dft, integral_image, integral_image_sq, neuron_response);
			response = response + neuron_response;
		}
	}

	int s_to_use = -1;

	// Find the matching sigma
	for(size_t i=0; i < window_sizes.size(); ++i)
	{
		if(window_sizes[i] == response_height)
		{
			// Found the correct sigma
			s_to_use = i;			
			break;
		}
	}

	cv::Mat_<float> resp_vec_f = response.reshape(1, response_height * response_width);

	cv::Mat out = Sigmas[s_to_use] * resp_vec_f;
	
	response = out.reshape(1, response_height);

	// Making sure the response does not have negative numbers
	double min;

	minMaxIdx(response, &min, 0);
	if(min < 0)
	{
		response = response - min;
	}

}

//===========================================================================
void CCNF_patch_expert::ResponseOB(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response)
{

	int response_height = area_of_interest.rows - height + 1;
	int response_width = area_of_interest.cols - width + 1;

	if (response.rows != response_height || response.cols != response_width)
	{
		response.create(response_height, response_width);
	}

	response.setTo(0);
	if (neurons.size() == 0)
	{
		return;
	}

	// the placeholder for the column normalized representation of the image, don't get recalculated for every response
	cv::Mat_<float> normalized_input;

	im2colContrastNorm(area_of_interest, neurons[0].weights.cols, neurons[0].weights.rows, normalized_input);
	normalized_input = normalized_input.t();

	// the placeholder for the DFT of the image, the integral image, and squared integral image so they don't get recalculated for every response
	cv::Mat_<double> area_of_interest_dft;
	cv::Mat integral_image, integral_image_sq;

	cv::Mat_<float> neuron_response;

	
	int h = area_of_interest.rows - neurons[0].weights.rows + 1;
	int w = area_of_interest.cols - neurons[0].weights.cols + 1;


	cv::Mat_<float> neuron_resp_full(weight_matrix.rows, normalized_input.cols, 0.0f);
	// Perform matrix multiplication in OpenBLAS (fortran call)
	float alpha1 = 1.0;
	float beta1 = 0.0;
	// TODO this should be a macro
	sgemm_("N", "N", &normalized_input.cols, &weight_matrix.rows, &weight_matrix.cols, &alpha1, (float*)normalized_input.data, &normalized_input.cols, (float*)weight_matrix.data, &weight_matrix.cols, &beta1, (float*)neuron_resp_full.data, &normalized_input.cols);

	// Above is a faster version of this
	//cv::Mat_<float> neuron_resp_full = this->weight_matrix * normalized_input;

	for (size_t i = 0; i < neurons.size(); i++)
	{
		if (neurons[i].alpha > 1e-4)
		{
			cv::MatIterator_<float> p = response.begin();

			cv::Mat_<float> rel_row = neuron_resp_full.row(i);// .clone();
			cv::MatIterator_<float> q1 = rel_row.begin(); // respone for each pixel
			cv::MatIterator_<float> q2 = rel_row.end();

			// the logistic function (sigmoid) applied to the response
			while (q1 != q2)
			{
				*p++ += (2 * neurons[i].alpha) * 1.0 / (1.0 + exp(-(*q1++ * neurons[i].norm_weights + neurons[i].bias)));
			}
		}
	}
	response = response.t();

	int s_to_use = -1;

	// Find the matching sigma
	for (size_t i = 0; i < window_sizes.size(); ++i)
	{
		if (window_sizes[i] == response_height)
		{
			// Found the correct sigma
			s_to_use = i;
			break;
		}
	}

	cv::Mat_<float> resp_vec_f = response.reshape(1, response_height * response_width);

	cv::Mat_<float> out(Sigmas[s_to_use].rows, resp_vec_f.cols, 0.0f);
	
	// Perform matrix multiplication in OpenBLAS (fortran call)
	alpha1 = 1.0;
	beta1 = 0.0;
	// TODO this should be a macro
	sgemm_("N", "N", &resp_vec_f.cols, &Sigmas[s_to_use].rows, &Sigmas[s_to_use].cols, &alpha1, (float*)resp_vec_f.data, &resp_vec_f.cols, (float*)Sigmas[s_to_use].data, &Sigmas[s_to_use].cols, &beta1, (float*)out.data, &resp_vec_f.cols);

	// Above is a faster version of this
	//cv::Mat out = Sigmas[s_to_use] * resp_vec_f;

	response = out.reshape(1, response_height);

	// Making sure the response does not have negative numbers
	double min;

	minMaxIdx(response, &min, 0);
	if (min < 0)
	{
		response = response - min;
	}

}
