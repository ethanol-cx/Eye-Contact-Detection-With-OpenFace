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

#include <LandmarkDetectorUtils.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

// Boost includes
#include <filesystem.hpp>
#include <filesystem/fstream.hpp>

using namespace boost::filesystem;

using namespace std;

namespace LandmarkDetector
{

	// For subpixel accuracy drawing
	const int draw_shiftbits = 4;
	const int draw_multiplier = 1 << 4;


	// Useful utility for creating directories for storing the output files
	void create_directory_from_file(string output_path)
	{

		// Creating the right directory structure

		// First get rid of the file
		auto p = path(path(output_path).parent_path());

		if (!p.empty() && !boost::filesystem::exists(p))
		{
			bool success = boost::filesystem::create_directories(p);
			if (!success)
			{
				cout << "Failed to create a directory... " << p.string() << endl;
			}
		}
	}

	// Useful utility for creating directories for storing the output files
	void create_directories(string output_path)
	{

		// Creating the right directory structure

		// First get rid of the file
		auto p = path(output_path);

		if (!p.empty() && !boost::filesystem::exists(p))
		{
			bool success = boost::filesystem::create_directories(p);
			if (!success)
			{
				cout << "Failed to create a directory... " << p.string() << endl;
			}
		}
	}

	// Extracting the following command line arguments -f, -op, -of, -ov (and possible ordered repetitions)
	void get_video_input_output_params(vector<string> &input_video_files, vector<string> &output_files, vector<string> &output_video_files, string& output_codec, vector<string> &arguments)
	{
		bool* valid = new bool[arguments.size()];

		for (size_t i = 0; i < arguments.size(); ++i)
		{
			valid[i] = true;
		}

		// By default use DIVX codec
		output_codec = "DIVX";

		string input_root = "";
		string output_root = "";

		string separator = string(1, boost::filesystem::path::preferred_separator);

		// First check if there is a root argument (so that videos and outputs could be defined more easilly)
		for (size_t i = 0; i < arguments.size(); ++i)
		{
			if (arguments[i].compare("-root") == 0)
			{
				input_root = arguments[i + 1] + separator;
				output_root = arguments[i + 1] + separator;

				// Add the / or \ to the directory
				i++;
			}
			if (arguments[i].compare("-inroot") == 0)
			{
				input_root = arguments[i + 1] + separator;
				i++;
			}
			if (arguments[i].compare("-outroot") == 0)
			{
				output_root = arguments[i + 1] + separator;
				i++;
			}
		}

		for (size_t i = 0; i < arguments.size(); ++i)
		{
			if (arguments[i].compare("-f") == 0)
			{
				input_video_files.push_back(input_root + arguments[i + 1]);
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
			else if (arguments[i].compare("-of") == 0)
			{
				output_files.push_back(output_root + arguments[i + 1]);
				create_directory_from_file(output_root + arguments[i + 1]);
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
			else if (arguments[i].compare("-ov") == 0)
			{
				output_video_files.push_back(output_root + arguments[i + 1]);
				create_directory_from_file(output_root + arguments[i + 1]);
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
			else if (arguments[i].compare("-oc") == 0)
			{
				if (arguments[i + 1].length() == 4)
					output_codec = arguments[i + 1];
			}
		}

		for (int i = arguments.size() - 1; i >= 0; --i)
		{
			if (!valid[i])
			{
				arguments.erase(arguments.begin() + i);
			}
		}

	}

	void get_camera_params(int &device, float &fx, float &fy, float &cx, float &cy, vector<string> &arguments)
	{
		bool* valid = new bool[arguments.size()];

		for (size_t i = 0; i < arguments.size(); ++i)
		{
			valid[i] = true;
			if (arguments[i].compare("-fx") == 0)
			{
				stringstream data(arguments[i + 1]);
				data >> fx;
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
			else if (arguments[i].compare("-fy") == 0)
			{
				stringstream data(arguments[i + 1]);
				data >> fy;
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
			else if (arguments[i].compare("-cx") == 0)
			{
				stringstream data(arguments[i + 1]);
				data >> cx;
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
			else if (arguments[i].compare("-cy") == 0)
			{
				stringstream data(arguments[i + 1]);
				data >> cy;
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
			else if (arguments[i].compare("-device") == 0)
			{
				stringstream data(arguments[i + 1]);
				data >> device;
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
		}

		for (int i = arguments.size() - 1; i >= 0; --i)
		{
			if (!valid[i])
			{
				arguments.erase(arguments.begin() + i);
			}
		}
	}

	void get_image_input_output_params(vector<string> &input_image_files, vector<string> &output_feature_files, vector<string> &output_3D_files, vector<string> &output_image_files,
		vector<cv::Rect_<double>> &input_bounding_boxes, vector<string> &arguments)
	{
		bool* valid = new bool[arguments.size()];

		string out_pts_dir, out_pose_dir, out_img_dir;

		string input_root = "";
		string output_root = "";

		string separator = string(1, boost::filesystem::path::preferred_separator);

		// First check if there is a root argument (so that videos and outputs could be defined more easilly)
		for (size_t i = 0; i < arguments.size(); ++i)
		{
			if (arguments[i].compare("-root") == 0)
			{
				input_root = arguments[i + 1] + separator;
				output_root = arguments[i + 1] + separator;
				i++;
			}
			if (arguments[i].compare("-inroot") == 0)
			{
				input_root = arguments[i + 1] + separator;
				i++;
			}
			if (arguments[i].compare("-outroot") == 0)
			{
				output_root = arguments[i + 1] + separator;
				i++;
			}
		}

		for (size_t i = 0; i < arguments.size(); ++i)
		{
			valid[i] = true;
			if (arguments[i].compare("-f") == 0)
			{
				input_image_files.push_back(input_root + arguments[i + 1]);
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}

			else if (arguments[i].compare("-fdir") == 0)
			{
				// parse the -fdir directory by reading in all of the .png and .jpg files in it
				path image_directory(arguments[i + 1]);

				try
				{
					// does the file exist and is it a directory
					if (exists(image_directory) && is_directory(image_directory))
					{

						vector<path> file_in_directory;
						copy(directory_iterator(image_directory), directory_iterator(), back_inserter(file_in_directory));

						// Sort the images in the directory first
						sort(file_in_directory.begin(), file_in_directory.end());

						for (vector<path>::const_iterator file_iterator(file_in_directory.begin()); file_iterator != file_in_directory.end(); ++file_iterator)
						{
							// Possible image extension .jpg and .png
							if (file_iterator->extension().string().compare(".jpg") == 0 || file_iterator->extension().string().compare(".png") == 0 || file_iterator->extension().string().compare(".bmp") == 0)
							{


								input_image_files.push_back(file_iterator->string());

								// If there exists a .txt file corresponding to the image, it is assumed that it contains a bounding box definition for a face
								// [minx, miny, maxx, maxy]
								path current_file = *file_iterator;
								path bbox = current_file.replace_extension("txt");

								// If there is a bounding box file push it to the list of bounding boxes
								if (exists(bbox))
								{

									std::ifstream in_bbox(bbox.string().c_str(), ios_base::in);

									double min_x, min_y, max_x, max_y;

									in_bbox >> min_x >> min_y >> max_x >> max_y;

									in_bbox.close();

									input_bounding_boxes.push_back(cv::Rect_<double>(min_x, min_y, max_x - min_x, max_y - min_y));
								}
							}
						}
					}
				}
				catch (const filesystem_error& ex)
				{
					cout << ex.what() << '\n';
				}

				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
			else if (arguments[i].compare("-ofdir") == 0)
			{
				out_pts_dir = arguments[i + 1];
				create_directories(out_pts_dir);
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
			else if (arguments[i].compare("-opdir") == 0)
			{
				out_pose_dir = arguments[i + 1];
				create_directories(out_pose_dir);
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
			else if (arguments[i].compare("-oidir") == 0)
			{
				out_img_dir = arguments[i + 1];
				create_directories(out_img_dir);
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
			else if (arguments[i].compare("-op") == 0)
			{
				output_3D_files.push_back(output_root + arguments[i + 1]);
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
			else if (arguments[i].compare("-of") == 0)
			{
				output_feature_files.push_back(output_root + arguments[i + 1]);
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
			else if (arguments[i].compare("-oi") == 0)
			{
				output_image_files.push_back(output_root + arguments[i + 1]);
				valid[i] = false;
				valid[i + 1] = false;
				i++;
			}
		}

		// If any output directories are defined populate them based on image names
		if (!out_img_dir.empty())
		{
			for (size_t i = 0; i < input_image_files.size(); ++i)
			{
				path image_loc(input_image_files[i]);

				path fname = image_loc.filename();
				fname = fname.replace_extension("bmp");
				output_image_files.push_back(out_img_dir + "/" + fname.string());

			}
			if (!input_image_files.empty())
			{
				create_directory_from_file(output_image_files[0]);
			}
		}

		if (!out_pts_dir.empty())
		{
			for (size_t i = 0; i < input_image_files.size(); ++i)
			{
				path image_loc(input_image_files[i]);

				path fname = image_loc.filename();
				fname = fname.replace_extension("pts");
				output_feature_files.push_back(out_pts_dir + "/" + fname.string());
			}
			create_directory_from_file(output_feature_files[0]);
		}

		if (!out_pose_dir.empty())
		{
			for (size_t i = 0; i < input_image_files.size(); ++i)
			{
				path image_loc(input_image_files[i]);

				path fname = image_loc.filename();
				fname = fname.replace_extension("pose");
				output_3D_files.push_back(out_pose_dir + "/" + fname.string());
			}
			create_directory_from_file(output_3D_files[0]);
		}

		// Make sure the same number of images and bounding boxes is present, if any bounding boxes are defined
		if (input_bounding_boxes.size() > 0)
		{
			if(input_bounding_boxes.size() != input_image_files.size())
			{
				cout << "Warning, the input number of images does not match the input number of bounding boxes\n" << endl;
			}
		}

		// Clear up the argument list
		for (int i = arguments.size() - 1; i >= 0; --i)
		{
			if (!valid[i])
			{
				arguments.erase(arguments.begin() + i);
			}
		}

	}

	//===========================================================================
	// Fast patch expert response computation (linear model across a ROI) using normalised cross-correlation
	//===========================================================================

	void crossCorr_m(const cv::Mat_<float>& img, cv::Mat_<double>& img_dft, const cv::Mat_<float>& _templ, map<int, cv::Mat_<double> >& _templ_dfts, cv::Mat_<float>& corr)
	{
		// Our model will always be under min block size so can ignore this
		//const double blockScale = 4.5;
		//const int minBlockSize = 256;

		int maxDepth = CV_64F;

		cv::Size dftsize;

		dftsize.width = cv::getOptimalDFTSize(corr.cols + _templ.cols - 1);
		dftsize.height = cv::getOptimalDFTSize(corr.rows + _templ.rows - 1);

		// Compute block size
		cv::Size blocksize;
		blocksize.width = dftsize.width - _templ.cols + 1;
		blocksize.width = MIN(blocksize.width, corr.cols);
		blocksize.height = dftsize.height - _templ.rows + 1;
		blocksize.height = MIN(blocksize.height, corr.rows);

		cv::Mat_<double> dftTempl;

		// if this has not been precomputed, precompute it, otherwise use it
		if (_templ_dfts.find(dftsize.width) == _templ_dfts.end())
		{
			dftTempl.create(dftsize.height, dftsize.width);

			cv::Mat_<float> src = _templ;

			cv::Mat_<double> dst(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));

			cv::Mat_<double> dst1(dftTempl, cv::Rect(0, 0, _templ.cols, _templ.rows));

			if (dst1.data != src.data)
				src.convertTo(dst1, dst1.depth());

			if (dst.cols > _templ.cols)
			{
				cv::Mat_<double> part(dst, cv::Range(0, _templ.rows), cv::Range(_templ.cols, dst.cols));
				part.setTo(0);
			}

			// Perform DFT of the template
			dft(dst, dst, 0, _templ.rows);

			_templ_dfts[dftsize.width] = dftTempl;

		}
		else
		{
			// use the precomputed version
			dftTempl = _templ_dfts.find(dftsize.width)->second;
		}

		cv::Size bsz(std::min(blocksize.width, corr.cols), std::min(blocksize.height, corr.rows));
		cv::Mat src;

		cv::Mat cdst(corr, cv::Rect(0, 0, bsz.width, bsz.height));

		cv::Mat_<double> dftImg;

		if (img_dft.empty())
		{
			dftImg.create(dftsize);
			dftImg.setTo(0.0);

			cv::Size dsz(bsz.width + _templ.cols - 1, bsz.height + _templ.rows - 1);

			int x2 = std::min(img.cols, dsz.width);
			int y2 = std::min(img.rows, dsz.height);

			cv::Mat src0(img, cv::Range(0, y2), cv::Range(0, x2));
			cv::Mat dst(dftImg, cv::Rect(0, 0, dsz.width, dsz.height));
			cv::Mat dst1(dftImg, cv::Rect(0, 0, x2, y2));

			src = src0;

			if (dst1.data != src.data)
				src.convertTo(dst1, dst1.depth());

			dft(dftImg, dftImg, 0, dsz.height);
			img_dft = dftImg.clone();
		}

		cv::Mat dftTempl1(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));
		cv::mulSpectrums(img_dft, dftTempl1, dftImg, 0, true);
		cv::dft(dftImg, dftImg, cv::DFT_INVERSE + cv::DFT_SCALE, bsz.height);

		src = dftImg(cv::Rect(0, 0, bsz.width, bsz.height));

		src.convertTo(cdst, CV_32F);

	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////

	void matchTemplate_m(const cv::Mat_<float>& input_img, cv::Mat_<double>& img_dft, cv::Mat& _integral_img, cv::Mat& _integral_img_sq, const cv::Mat_<float>&  templ, map<int, cv::Mat_<double> >& templ_dfts, cv::Mat_<float>& result, int method)
	{

		int numType = method == CV_TM_CCORR || method == CV_TM_CCORR_NORMED ? 0 :
			method == CV_TM_CCOEFF || method == CV_TM_CCOEFF_NORMED ? 1 : 2;
		bool isNormed = method == CV_TM_CCORR_NORMED ||
			method == CV_TM_SQDIFF_NORMED ||
			method == CV_TM_CCOEFF_NORMED;

		// Assume result is defined properly
		if (result.empty())
		{
			cv::Size corrSize(input_img.cols - templ.cols + 1, input_img.rows - templ.rows + 1);
			result.create(corrSize);
		}
		LandmarkDetector::crossCorr_m(input_img, img_dft, templ, templ_dfts, result);

		if (method == CV_TM_CCORR)
			return;

		double invArea = 1. / ((double)templ.rows * templ.cols);

		cv::Mat sum, sqsum;
		cv::Scalar templMean, templSdv;
		double *q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;
		double templNorm = 0, templSum2 = 0;

		if (method == CV_TM_CCOEFF)
		{
			// If it has not been precomputed compute it now
			if (_integral_img.empty())
			{
				integral(input_img, _integral_img, CV_64F);
			}
			sum = _integral_img;

			templMean = cv::mean(templ);
		}
		else
		{
			// If it has not been precomputed compute it now
			if (_integral_img.empty())
			{
				integral(input_img, _integral_img, _integral_img_sq, CV_64F);
			}

			sum = _integral_img;
			sqsum = _integral_img_sq;

			meanStdDev(templ, templMean, templSdv);

			templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];

			if (templNorm < DBL_EPSILON && method == CV_TM_CCOEFF_NORMED)
			{
				result.setTo(1.0);
				return;
			}

			templSum2 = templNorm + templMean[0] * templMean[0] + templMean[1] * templMean[1] + templMean[2] * templMean[2] + templMean[3] * templMean[3];

			if (numType != 1)
			{
				templMean = cv::Scalar::all(0);
				templNorm = templSum2;
			}

			templSum2 /= invArea;
			templNorm = std::sqrt(templNorm);
			templNorm /= std::sqrt(invArea); // care of accuracy here

			q0 = (double*)sqsum.data;
			q1 = q0 + templ.cols;
			q2 = (double*)(sqsum.data + templ.rows*sqsum.step);
			q3 = q2 + templ.cols;
		}

		double* p0 = (double*)sum.data;
		double* p1 = p0 + templ.cols;
		double* p2 = (double*)(sum.data + templ.rows*sum.step);
		double* p3 = p2 + templ.cols;

		int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
		int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;

		int i, j;

		for (i = 0; i < result.rows; i++)
		{
			float* rrow = result.ptr<float>(i);
			int idx = i * sumstep;
			int idx2 = i * sqstep;

			for (j = 0; j < result.cols; j++, idx += 1, idx2 += 1)
			{
				double num = rrow[j], t;
				double wndMean2 = 0, wndSum2 = 0;

				if (numType == 1)
				{

					t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
					wndMean2 += t*t;
					num -= t*templMean[0];

					wndMean2 *= invArea;
				}

				if (isNormed || numType == 2)
				{

					t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
					wndSum2 += t;

					if (numType == 2)
					{
						num = wndSum2 - 2 * num + templSum2;
						num = MAX(num, 0.);
					}
				}

				if (isNormed)
				{
					t = std::sqrt(MAX(wndSum2 - wndMean2, 0))*templNorm;
					if (fabs(num) < t)
						num /= t;
					else if (fabs(num) < t*1.125)
						num = num > 0 ? 1 : -1;
					else
						num = method != CV_TM_SQDIFF_NORMED ? 0 : 1;
				}

				rrow[j] = (float)num;
			}
		}
	}


	//===========================================================================
	// Point set and landmark manipulation functions
	//===========================================================================
	// Using Kabsch's algorithm for aligning shapes
	//This assumes that align_from and align_to are already mean normalised
	cv::Matx22d AlignShapesKabsch2D(const cv::Mat_<double>& align_from, const cv::Mat_<double>& align_to)
	{

		cv::SVD svd(align_from.t() * align_to);

		// make sure no reflection is there
		// corr ensures that we do only rotaitons and not reflections
		double d = cv::determinant(svd.vt.t() * svd.u.t());

		cv::Matx22d corr = cv::Matx22d::eye();
		if (d > 0)
		{
			corr(1, 1) = 1;
		}
		else
		{
			corr(1, 1) = -1;
		}

		cv::Matx22d R;
		cv::Mat(svd.vt.t()*cv::Mat(corr)*svd.u.t()).copyTo(R);

		return R;
	}

	cv::Matx22f AlignShapesKabsch2D_f(const cv::Mat_<float>& align_from, const cv::Mat_<float>& align_to)
	{

		cv::SVD svd(align_from.t() * align_to);

		// make sure no reflection is there
		// corr ensures that we do only rotaitons and not reflections
		float d = cv::determinant(svd.vt.t() * svd.u.t());

		cv::Matx22f corr = cv::Matx22f::eye();
		if (d > 0)
		{
			corr(1, 1) = 1;
		}
		else
		{
			corr(1, 1) = -1;
		}

		cv::Matx22f R;
		cv::Mat(svd.vt.t()*cv::Mat(corr)*svd.u.t()).copyTo(R);

		return R;
	}

	//=============================================================================
	// Basically Kabsch's algorithm but also allows the collection of points to be different in scale from each other
	cv::Matx22d AlignShapesWithScale(cv::Mat_<double>& src, cv::Mat_<double> dst)
	{
		int n = src.rows;

		// First we mean normalise both src and dst
		double mean_src_x = cv::mean(src.col(0))[0];
		double mean_src_y = cv::mean(src.col(1))[0];

		double mean_dst_x = cv::mean(dst.col(0))[0];
		double mean_dst_y = cv::mean(dst.col(1))[0];

		cv::Mat_<double> src_mean_normed = src.clone();
		src_mean_normed.col(0) = src_mean_normed.col(0) - mean_src_x;
		src_mean_normed.col(1) = src_mean_normed.col(1) - mean_src_y;

		cv::Mat_<double> dst_mean_normed = dst.clone();
		dst_mean_normed.col(0) = dst_mean_normed.col(0) - mean_dst_x;
		dst_mean_normed.col(1) = dst_mean_normed.col(1) - mean_dst_y;

		// Find the scaling factor of each
		cv::Mat src_sq;
		cv::pow(src_mean_normed, 2, src_sq);

		cv::Mat dst_sq;
		cv::pow(dst_mean_normed, 2, dst_sq);

		double s_src = sqrt(cv::sum(src_sq)[0] / n);
		double s_dst = sqrt(cv::sum(dst_sq)[0] / n);

		src_mean_normed = src_mean_normed / s_src;
		dst_mean_normed = dst_mean_normed / s_dst;

		double s = s_dst / s_src;

		// Get the rotation
		cv::Matx22d R = AlignShapesKabsch2D(src_mean_normed, dst_mean_normed);

		cv::Matx22d	A;
		cv::Mat(s * R).copyTo(A);

		cv::Mat_<double> aligned = (cv::Mat(cv::Mat(A) * src.t())).t();
		cv::Mat_<double> offset = dst - aligned;

		double t_x = cv::mean(offset.col(0))[0];
		double t_y = cv::mean(offset.col(1))[0];

		return A;

	}

	cv::Matx22f AlignShapesWithScale_f(cv::Mat_<float>& src, cv::Mat_<float> dst)
	{
		int n = src.rows;

		// First we mean normalise both src and dst
		float mean_src_x = cv::mean(src.col(0))[0];
		float mean_src_y = cv::mean(src.col(1))[0];

		float mean_dst_x = cv::mean(dst.col(0))[0];
		float mean_dst_y = cv::mean(dst.col(1))[0];

		cv::Mat_<float> src_mean_normed = src.clone();
		src_mean_normed.col(0) = src_mean_normed.col(0) - mean_src_x;
		src_mean_normed.col(1) = src_mean_normed.col(1) - mean_src_y;

		cv::Mat_<float> dst_mean_normed = dst.clone();
		dst_mean_normed.col(0) = dst_mean_normed.col(0) - mean_dst_x;
		dst_mean_normed.col(1) = dst_mean_normed.col(1) - mean_dst_y;

		// Find the scaling factor of each
		cv::Mat src_sq;
		cv::pow(src_mean_normed, 2, src_sq);

		cv::Mat dst_sq;
		cv::pow(dst_mean_normed, 2, dst_sq);

		float s_src = sqrt(cv::sum(src_sq)[0] / n);
		float s_dst = sqrt(cv::sum(dst_sq)[0] / n);

		src_mean_normed = src_mean_normed / s_src;
		dst_mean_normed = dst_mean_normed / s_dst;

		float s = s_dst / s_src;

		// Get the rotation
		cv::Matx22f R = AlignShapesKabsch2D_f(src_mean_normed, dst_mean_normed);

		cv::Matx22f	A;
		cv::Mat(s * R).copyTo(A);

		cv::Mat_<float> aligned = (cv::Mat(cv::Mat(A) * src.t())).t();
		cv::Mat_<float> offset = dst - aligned;

		float t_x = cv::mean(offset.col(0))[0];
		float t_y = cv::mean(offset.col(1))[0];

		return A;

	}

	// Useful utility for grabing a bounding box around a set of 2D landmarks (as a 1D 2n x 1 vector of xs followed by doubles or as an n x 2 vector)
	void ExtractBoundingBox(const cv::Mat_<float>& landmarks, float &min_x, float &max_x, float &min_y, float &max_y)
	{

		if (landmarks.cols == 1)
		{
			int n = landmarks.rows / 2;
			cv::MatConstIterator_<float> landmarks_it = landmarks.begin();

			for (int i = 0; i < n; ++i)
			{
				float val = *landmarks_it++;
				
				if (i == 0 || val < min_x)
					min_x = val;

				if (i == 0 || val > max_x)
					max_x = val;

			}

			for (int i = 0; i < n; ++i)
			{
				float val = *landmarks_it++;

				if (i == 0 || val < min_y)
					min_y = val;

				if (i == 0 || val > max_y)
					max_y = val;

			}
		}
		else
		{
			int n = landmarks.rows;
			for (int i = 0; i < n; ++i)
			{
				float val_x = landmarks.at<float>(i, 0);
				float val_y = landmarks.at<float>(i, 0);

				if (i == 0 || val_x < min_x)
					min_x = val_x;

				if (i == 0 || val_x > max_x)
					max_x = val_x;

				if (i == 0 || val_y < min_y)
					min_y = val_y;

				if (i == 0 || val_y > max_y)
					max_y = val_y;

			}

		}


	}

	//===========================================================================
	// Visualisation functions
	//===========================================================================
	void Project(cv::Mat_<float>& dest, const cv::Mat_<float>& mesh, float fx, float fy, float cx, float cy)
	{
		dest = cv::Mat_<float>(mesh.rows, 2, 0.0);

		int num_points = mesh.rows;

		float X, Y, Z;


		cv::Mat_<float>::const_iterator mData = mesh.begin();
		cv::Mat_<float>::iterator projected = dest.begin();

		for (int i = 0; i < num_points; i++)
		{
			// Get the points
			X = *(mData++);
			Y = *(mData++);
			Z = *(mData++);

			float x;
			float y;

			// if depth is 0 the projection is different
			if (Z != 0)
			{
				x = ((X * fx / Z) + cx);
				y = ((Y * fy / Z) + cy);
			}
			else
			{
				x = X;
				y = Y;
			}

			// Project and store in dest matrix
			(*projected++) = x;
			(*projected++) = y;
		}

	}

	void DrawBox(cv::Mat image, cv::Vec6f pose, cv::Scalar color, int thickness, float fx, float fy, float cx, float cy)
	{
		float boxVerts[] = { -1, 1, -1,
			1, 1, -1,
			1, 1, 1,
			-1, 1, 1,
			1, -1, 1,
			1, -1, -1,
			-1, -1, -1,
			-1, -1, 1 };

		vector<std::pair<int, int>> edges;
		edges.push_back(pair<int, int>(0, 1));
		edges.push_back(pair<int, int>(1, 2));
		edges.push_back(pair<int, int>(2, 3));
		edges.push_back(pair<int, int>(0, 3));
		edges.push_back(pair<int, int>(2, 4));
		edges.push_back(pair<int, int>(1, 5));
		edges.push_back(pair<int, int>(0, 6));
		edges.push_back(pair<int, int>(3, 7));
		edges.push_back(pair<int, int>(6, 5));
		edges.push_back(pair<int, int>(5, 4));
		edges.push_back(pair<int, int>(4, 7));
		edges.push_back(pair<int, int>(7, 6));

		// The size of the head is roughly 200mm x 200mm x 200mm
		cv::Mat_<float> box = cv::Mat(8, 3, CV_32F, boxVerts).clone() * 100;

		cv::Matx33f rot = LandmarkDetector::Euler2RotationMatrix(cv::Vec3f((float)pose[3], (float)pose[4], (float)pose[5]));
		cv::Mat_<float> rotBox;

		// Rotate the box
		rotBox = cv::Mat(rot) * box.t();
		rotBox = rotBox.t();

		// Move the bounding box to head position
		rotBox.col(0) = rotBox.col(0) + pose[0];
		rotBox.col(1) = rotBox.col(1) + pose[1];
		rotBox.col(2) = rotBox.col(2) + pose[2];

		// draw the lines
		cv::Mat_<float> rotBoxProj;
		Project(rotBoxProj, rotBox, fx, fy, cx, cy);

		cv::Rect image_rect(0, 0, image.cols * draw_multiplier, image.rows * draw_multiplier);

		for (size_t i = 0; i < edges.size(); ++i)
		{
			cv::Mat_<float> begin;
			cv::Mat_<float> end;

			rotBoxProj.row(edges[i].first).copyTo(begin);
			rotBoxProj.row(edges[i].second).copyTo(end);


			cv::Point p1(cvRound(begin.at<float>(0) * (float)draw_multiplier), cvRound(begin.at<float>(1) * (float)draw_multiplier));
			cv::Point p2(cvRound(end.at<float>(0) * (float)draw_multiplier), cvRound(end.at<float>(1) * (float)draw_multiplier));

			// Only draw the line if one of the points is inside the image
			if (p1.inside(image_rect) || p2.inside(image_rect))
			{
				cv::line(image, p1, p2, color, thickness, CV_AA, draw_shiftbits);
			}

		}

	}

	vector<std::pair<cv::Point2f, cv::Point2f>> CalculateBox(cv::Vec6f pose, float fx, float fy, float cx, float cy)
	{
		float boxVerts[] = { -1, 1, -1,
			1, 1, -1,
			1, 1, 1,
			-1, 1, 1,
			1, -1, 1,
			1, -1, -1,
			-1, -1, -1,
			-1, -1, 1 };

		vector<std::pair<int, int>> edges;
		edges.push_back(pair<int, int>(0, 1));
		edges.push_back(pair<int, int>(1, 2));
		edges.push_back(pair<int, int>(2, 3));
		edges.push_back(pair<int, int>(0, 3));
		edges.push_back(pair<int, int>(2, 4));
		edges.push_back(pair<int, int>(1, 5));
		edges.push_back(pair<int, int>(0, 6));
		edges.push_back(pair<int, int>(3, 7));
		edges.push_back(pair<int, int>(6, 5));
		edges.push_back(pair<int, int>(5, 4));
		edges.push_back(pair<int, int>(4, 7));
		edges.push_back(pair<int, int>(7, 6));

		// The size of the head is roughly 200mm x 200mm x 200mm
		cv::Mat_<float> box = cv::Mat(8, 3, CV_32F, boxVerts).clone() * 100;

		cv::Matx33f rot = LandmarkDetector::Euler2RotationMatrix(cv::Vec3d((float) pose[3], (float)pose[4], (float)pose[5]));
		cv::Mat_<float> rotBox;

		// Rotate the box
		rotBox = cv::Mat(rot) * box.t();
		rotBox = rotBox.t();

		// Move the bounding box to head position
		rotBox.col(0) = rotBox.col(0) + pose[0];
		rotBox.col(1) = rotBox.col(1) + pose[1];
		rotBox.col(2) = rotBox.col(2) + pose[2];

		// draw the lines
		cv::Mat_<float> rotBoxProj;
		Project(rotBoxProj, rotBox, fx, fy, cx, cy);

		vector<std::pair<cv::Point2f, cv::Point2f>> lines;

		for (size_t i = 0; i < edges.size(); ++i)
		{
			cv::Mat_<float> begin;
			cv::Mat_<float> end;

			rotBoxProj.row(edges[i].first).copyTo(begin);
			rotBoxProj.row(edges[i].second).copyTo(end);

			cv::Point2d p1(begin.at<float>(0), begin.at<float>(1));
			cv::Point2d p2(end.at<float>(0), end.at<float>(1));

			lines.push_back(pair<cv::Point2f, cv::Point2f>(p1, p2));

		}

		return lines;
	}

	void DrawBox(vector<pair<cv::Point, cv::Point>> lines, cv::Mat image, cv::Scalar color, int thickness)
	{
		cv::Rect image_rect(0, 0, image.cols, image.rows);

		for (size_t i = 0; i < lines.size(); ++i)
		{
			cv::Point p1 = lines.at(i).first;
			cv::Point p2 = lines.at(i).second;
			// Only draw the line if one of the points is inside the image
			if (p1.inside(image_rect) || p2.inside(image_rect))
			{
				cv::line(image, p1, p2, color, thickness, CV_AA);
			}

		}

	}

	// Computing landmarks (to be drawn later possibly)
	vector<cv::Point2d> CalculateVisibleLandmarks(const cv::Mat_<float>& shape2D, const cv::Mat_<int>& visibilities)
	{
		int n = shape2D.rows / 2;
		vector<cv::Point2d> landmarks;

		for (int i = 0; i < n; ++i)
		{
			if (visibilities.at<int>(i))
			{
				cv::Point2d featurePoint(shape2D.at<float>(i), shape2D.at<float>(i + n));

				landmarks.push_back(featurePoint);
			}
		}

		return landmarks;
	}

	// Computing landmarks (to be drawn later possibly)
	vector<cv::Point2f> CalculateAllLandmarks(const cv::Mat_<float>& shape2D)
	{

		int n;
		vector<cv::Point2f> landmarks;

		if (shape2D.cols == 2)
		{
			n = shape2D.rows;
		}
		else if (shape2D.cols == 1)
		{
			n = shape2D.rows / 2;
		}

		for (int i = 0; i < n; ++i)
		{
			cv::Point2f featurePoint;
			if (shape2D.cols == 1)
			{
				featurePoint = cv::Point2f(shape2D.at<float>(i), shape2D.at<float>(i + n));
			}
			else
			{
				featurePoint = cv::Point2f(shape2D.at<float>(i, 0), shape2D.at<float>(i, 1));
			}

			landmarks.push_back(featurePoint);
		}

		return landmarks;
	}

	// Computing landmarks (to be drawn later possibly)
	vector<cv::Point2f> CalculateAllLandmarks(const CLNF& clnf_model)
	{
		return CalculateAllLandmarks(clnf_model.detected_landmarks);
	}

	// Computing landmarks (to be drawn later possibly)
	vector<cv::Point2d> CalculateVisibleLandmarks(const CLNF& clnf_model)
	{
		// If the detection was not successful no landmarks are visible
		if (clnf_model.detection_success)
		{
			int idx = clnf_model.patch_experts.GetViewIdx(clnf_model.params_global, 0);
			// Because we only draw visible points, need to find which points patch experts consider visible at a certain orientation
			return CalculateVisibleLandmarks(clnf_model.detected_landmarks, clnf_model.patch_experts.visibilities[0][idx]);
		}
		else
		{
			return vector<cv::Point2d>();
		}
	}

	// Computing eye landmarks (to be drawn later or in different interfaces)
	vector<cv::Point2d> CalculateVisibleEyeLandmarks(const CLNF& clnf_model)
	{

		vector<cv::Point2d> to_return;
		// If the model has hierarchical updates draw those too
		for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
		{

			if (clnf_model.hierarchical_model_names[i].compare("left_eye_28") == 0 ||
				clnf_model.hierarchical_model_names[i].compare("right_eye_28") == 0)
			{

				auto lmks = CalculateVisibleLandmarks(clnf_model.hierarchical_models[i]);
				for (auto lmk : lmks)
				{
					to_return.push_back(lmk);
				}
			}
		}
		return to_return;
	}
	// Computing the 3D eye landmarks
	vector<cv::Point3f> Calculate3DEyeLandmarks(const CLNF& clnf_model, float fx, float fy, float cx, float cy)
	{

		vector<cv::Point3f> to_return;

		for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
		{

			if (clnf_model.hierarchical_model_names[i].compare("left_eye_28") == 0 ||
				clnf_model.hierarchical_model_names[i].compare("right_eye_28") == 0)
			{

				auto lmks = clnf_model.hierarchical_models[i].GetShape(fx, fy, cx, cy);

				int num_landmarks = lmks.cols;

				for (int lmk = 0; lmk < num_landmarks; ++lmk)
				{
					cv::Point3f curr_lmk(lmks.at<float>(0, lmk), lmks.at<float>(1, lmk), lmks.at<float>(2, lmk));
					to_return.push_back(curr_lmk);
				}
			}
		}
		return to_return;
	}

	// Computing eye landmarks (to be drawn later or in different interfaces)
	vector<cv::Point2f> CalculateAllEyeLandmarks(const CLNF& clnf_model)
	{

		vector<cv::Point2f> to_return;
		// If the model has hierarchical updates draw those too
		for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
		{

			if (clnf_model.hierarchical_model_names[i].compare("left_eye_28") == 0 ||
				clnf_model.hierarchical_model_names[i].compare("right_eye_28") == 0)
			{

				auto lmks = CalculateAllLandmarks(clnf_model.hierarchical_models[i]);
				for (auto lmk : lmks)
				{
					to_return.push_back(lmk);
				}
			}
		}
		return to_return;
	}

	// Drawing landmarks on a face image
	void Draw(cv::Mat img, const cv::Mat_<float>& shape2D, const cv::Mat_<int>& visibilities)
	{
		int n = shape2D.rows / 2;


		// Drawing feature points
		if (n >= 66)
		{
			for (int i = 0; i < n; ++i)
			{
				if (visibilities.at<int>(i))
				{
					cv::Point featurePoint(cvRound(shape2D.at<float>(i) * (float)draw_multiplier), cvRound(shape2D.at<float>(i + n) * (float)draw_multiplier));

					// A rough heuristic for drawn point size
					int thickness = (int)std::ceil(3.0* ((double)img.cols) / 640.0);
					int thickness_2 = (int)std::ceil(1.0* ((double)img.cols) / 640.0);

					cv::circle(img, featurePoint, 1 * draw_multiplier, cv::Scalar(0, 0, 255), thickness, CV_AA, draw_shiftbits);
					cv::circle(img, featurePoint, 1 * draw_multiplier, cv::Scalar(255, 0, 0), thickness_2, CV_AA, draw_shiftbits);

				}
			}
		}
		else if (n == 28) // drawing eyes
		{
			for (int i = 0; i < n; ++i)
			{
				cv::Point featurePoint(cvRound(shape2D.at<float>(i) * (float)draw_multiplier), cvRound(shape2D.at<float>(i + n) * (float)draw_multiplier));

				// A rough heuristic for drawn point size
				int thickness = 1.0;
				int thickness_2 = 1.0;

				int next_point = i + 1;
				if (i == 7)
					next_point = 0;
				if (i == 19)
					next_point = 8;
				if (i == 27)
					next_point = 20;

				cv::Point nextFeaturePoint(cvRound(shape2D.at<float>(next_point) * (float)draw_multiplier), cvRound(shape2D.at<float>(next_point + n) * (float)draw_multiplier));
				if (i < 8 || i > 19)
					cv::line(img, featurePoint, nextFeaturePoint, cv::Scalar(255, 0, 0), thickness_2, CV_AA, draw_shiftbits);
				else
					cv::line(img, featurePoint, nextFeaturePoint, cv::Scalar(0, 0, 255), thickness_2, CV_AA, draw_shiftbits);


			}
		}
		else if (n == 6)
		{
			for (int i = 0; i < n; ++i)
			{
				cv::Point featurePoint(cvRound(shape2D.at<float>(i) * (float)draw_multiplier), cvRound(shape2D.at<float>(i + n) * (float)draw_multiplier));

				// A rough heuristic for drawn point size
				int thickness = 1.0;
				int thickness_2 = 1.0;

				int next_point = i + 1;
				if (i == 5)
					next_point = 0;

				cv::Point nextFeaturePoint(cvRound(shape2D.at<float>(next_point) * (float)draw_multiplier), cvRound(shape2D.at<float>(next_point + n) * (float)draw_multiplier));
				cv::line(img, featurePoint, nextFeaturePoint, cv::Scalar(255, 0, 0), thickness_2, CV_AA, draw_shiftbits);
			}
		}
	}

	// Drawing landmarks on a face image
	void Draw(cv::Mat img, const cv::Mat_<float>& shape2D)
	{

		int n;

		if (shape2D.cols == 2)
		{
			n = shape2D.rows;
		}
		else if (shape2D.cols == 1)
		{
			n = shape2D.rows / 2;
		}

		for (int i = 0; i < n; ++i)
		{
			cv::Point featurePoint;
			if (shape2D.cols == 1)
			{
				featurePoint = cv::Point(cvRound(shape2D.at<float>(i) * (float)draw_multiplier), cvRound(shape2D.at<float>(i + n) * (float)draw_multiplier));
			}
			else
			{
				featurePoint = cv::Point(cvRound(shape2D.at<float>(i, 0) * (float)draw_multiplier), cvRound(shape2D.at<float>(i, 1) * (float)draw_multiplier));
			}
			// A rough heuristic for drawn point size
			int thickness = (int)std::ceil(5.0* ((double)img.cols) / 640.0);
			int thickness_2 = (int)std::ceil(1.5* ((double)img.cols) / 640.0);

			cv::circle(img, featurePoint, 1 * draw_multiplier, cv::Scalar(0, 0, 255), thickness, CV_AA, draw_shiftbits);
			cv::circle(img, featurePoint, 1 * draw_multiplier, cv::Scalar(255, 0, 0), thickness_2, CV_AA, draw_shiftbits);

		}

	}

	// Drawing detected landmarks on a face image
	void Draw(cv::Mat img, const CLNF& clnf_model)
	{

		int idx = clnf_model.patch_experts.GetViewIdx(clnf_model.params_global, 0);

		// Because we only draw visible points, need to find which points patch experts consider visible at a certain orientation
		Draw(img, clnf_model.detected_landmarks, clnf_model.patch_experts.visibilities[0][idx]);

		// If the model has hierarchical updates draw those too
		for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
		{
			if (clnf_model.hierarchical_models[i].pdm.NumberOfPoints() != clnf_model.hierarchical_mapping[i].size())
			{
				Draw(img, clnf_model.hierarchical_models[i]);
			}
		}
	}

	void DrawLandmarks(cv::Mat img, vector<cv::Point> landmarks)
	{
		for (cv::Point p : landmarks)
		{

			// A rough heuristic for drawn point size
			int thickness = (int)std::ceil(5.0* ((double)img.cols) / 640.0);
			int thickness_2 = (int)std::ceil(1.5* ((double)img.cols) / 640.0);

			cv::circle(img, p, 1, cv::Scalar(0, 0, 255), thickness, CV_AA);
			cv::circle(img, p, 1, cv::Scalar(255, 0, 0), thickness_2, CV_AA);
		}

	}

	//===========================================================================
	// Angle representation conversion helpers
	//===========================================================================

	// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	cv::Matx33f Euler2RotationMatrix(const cv::Vec3f& eulerAngles)
	{
		cv::Matx33f rotation_matrix;

		float s1 = sin(eulerAngles[0]);
		float s2 = sin(eulerAngles[1]);
		float s3 = sin(eulerAngles[2]);

		float c1 = cos(eulerAngles[0]);
		float c2 = cos(eulerAngles[1]);
		float c3 = cos(eulerAngles[2]);

		rotation_matrix(0, 0) = c2 * c3;
		rotation_matrix(0, 1) = -c2 *s3;
		rotation_matrix(0, 2) = s2;
		rotation_matrix(1, 0) = c1 * s3 + c3 * s1 * s2;
		rotation_matrix(1, 1) = c1 * c3 - s1 * s2 * s3;
		rotation_matrix(1, 2) = -c2 * s1;
		rotation_matrix(2, 0) = s1 * s3 - c1 * c3 * s2;
		rotation_matrix(2, 1) = c3 * s1 + c1 * s2 * s3;
		rotation_matrix(2, 2) = c1 * c2;

		return rotation_matrix;
	}

	// Using the XYZ convention R = Rx * Ry * Rz, left-handed positive sign
	cv::Vec3f RotationMatrix2Euler(const cv::Matx33f& rotation_matrix)
	{
		float q0 = sqrt(1 + rotation_matrix(0, 0) + rotation_matrix(1, 1) + rotation_matrix(2, 2)) / 2.0f;
		float q1 = (rotation_matrix(2, 1) - rotation_matrix(1, 2)) / (4.0f*q0);
		float q2 = (rotation_matrix(0, 2) - rotation_matrix(2, 0)) / (4.0f*q0);
		float q3 = (rotation_matrix(1, 0) - rotation_matrix(0, 1)) / (4.0f*q0);

		float t1 = 2.0f * (q0*q2 + q1*q3);

		float yaw = asin(2.0 * (q0*q2 + q1*q3));
		float pitch = atan2(2.0 * (q0*q1 - q2*q3), q0*q0 - q1*q1 - q2*q2 + q3*q3);
		float roll = atan2(2.0 * (q0*q3 - q1*q2), q0*q0 + q1*q1 - q2*q2 - q3*q3);

		return cv::Vec3f(pitch, yaw, roll);
	}

	cv::Vec3f Euler2AxisAngle(const cv::Vec3f& euler)
	{
		cv::Matx33f rotMatrix = LandmarkDetector::Euler2RotationMatrix(euler);
		cv::Vec3f axis_angle;
		cv::Rodrigues(rotMatrix, axis_angle);
		return axis_angle;
	}

	cv::Vec3f AxisAngle2Euler(const cv::Vec3f& axis_angle)
	{
		cv::Matx33f rotation_matrix;
		cv::Rodrigues(axis_angle, rotation_matrix);
		return RotationMatrix2Euler(rotation_matrix);
	}

	cv::Matx33f AxisAngle2RotationMatrix(const cv::Vec3f& axis_angle)
	{
		cv::Matx33f rotation_matrix;
		cv::Rodrigues(axis_angle, rotation_matrix);
		return rotation_matrix;
	}

	cv::Vec3f RotationMatrix2AxisAngle(const cv::Matx33f& rotation_matrix)
	{
		cv::Vec3f axis_angle;
		cv::Rodrigues(rotation_matrix, axis_angle);
		return axis_angle;
	}
	//===========================================================================

	//============================================================================
	// Face detection helpers
	//============================================================================
	bool DetectFaces(vector<cv::Rect_<float> >& o_regions, const cv::Mat_<uchar>& intensity, float min_width, cv::Rect_<float> roi)
	{
		cv::CascadeClassifier classifier("./classifiers/haarcascade_frontalface_alt.xml");
		if (classifier.empty())
		{
			cout << "Couldn't load the Haar cascade classifier" << endl;
			return false;
		}
		else
		{
			return DetectFaces(o_regions, intensity, classifier, min_width, roi);
		}

	}

	bool DetectFaces(vector<cv::Rect_<float> >& o_regions, const cv::Mat_<uchar>& intensity, cv::CascadeClassifier& classifier, float min_width, cv::Rect_<float> roi)
	{

		vector<cv::Rect> face_detections;
		if (min_width == -1)
		{
			classifier.detectMultiScale(intensity, face_detections, 1.2, 2, 0, cv::Size(50, 50));
		}
		else
		{
			classifier.detectMultiScale(intensity, face_detections, 1.2, 2, 0, cv::Size(min_width, min_width));
		}

		// Convert from int bounding box do a double one with corrections
		for (size_t face = 0; face < o_regions.size(); ++face)
		{
			// OpenCV is overgenerous with face size and y location is off
			// CLNF detector expects the bounding box to encompass from eyebrow to chin in y, and from cheeck outline to cheeck outline in x, so we need to compensate

			// The scalings were learned using the Face Detections on LFPW, Helen, AFW and iBUG datasets, using ground truth and detections from openCV
			cv::Rect_<float> region;
			// Correct for scale
			region.width = face_detections[face].width * 0.8924f;
			region.height = face_detections[face].height * 0.8676f;

			// Move the face slightly to the right (as the width was made smaller)
			region.x = face_detections[face].x + 0.0578f * face_detections[face].width;
			// Shift face down as OpenCV Haar Cascade detects the forehead as well, and we're not interested
			region.y = face_detections[face].y + face_detections[face].height * 0.2166f;

			if (min_width != -1)
			{
				if (region.width < min_width || region.x < ((float)intensity.cols) * roi.x || region.y < ((float)intensity.cols) * roi.y ||
					region.x + region.width >((float)intensity.cols) * (roi.x + roi.width) || region.y + region.height >((float)intensity.rows) * (roi.y + roi.height))
					continue;
			}


			o_regions.push_back(region);
		}
		return o_regions.size() > 0;
	}

	bool DetectSingleFace(cv::Rect_<float>& o_region, const cv::Mat_<uchar>& intensity_image, cv::CascadeClassifier& classifier, cv::Point preference, float min_width, cv::Rect_<float> roi)
	{
		// The tracker can return multiple faces
		vector<cv::Rect_<float> > face_detections;

		bool detect_success = LandmarkDetector::DetectFaces(face_detections, intensity_image, classifier, min_width, roi);

		if (detect_success)
		{

			bool use_preferred = (preference.x != -1) && (preference.y != -1);

			if (face_detections.size() > 1)
			{
				// keep the closest one if preference point not set
				float best = -1;
				int bestIndex = -1;
				for (size_t i = 0; i < face_detections.size(); ++i)
				{
					float dist;
					bool better;

					if (use_preferred)
					{
						dist = sqrt((preference.x) * (face_detections[i].width / 2 + face_detections[i].x) +
							(preference.y) * (face_detections[i].height / 2 + face_detections[i].y));
						better = dist < best;
					}
					else
					{
						dist = face_detections[i].width;
						better = face_detections[i].width > best;
					}

					// Pick a closest face to preffered point or the biggest face
					if (i == 0 || better)
					{
						bestIndex = i;
						best = dist;
					}
				}

				o_region = face_detections[bestIndex];

			}
			else
			{
				o_region = face_detections[0];
			}

		}
		else
		{
			// if not detected
			o_region = cv::Rect_<float>(0, 0, 0, 0);
		}
		return detect_success;
	}

	bool DetectFacesHOG(vector<cv::Rect_<float> >& o_regions, const cv::Mat_<uchar>& intensity, std::vector<float>& confidences, float min_width, cv::Rect_<float> roi)
	{
		dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

		return DetectFacesHOG(o_regions, intensity, detector, confidences, min_width, roi);

	}

	bool DetectFacesHOG(vector<cv::Rect_<float> >& o_regions, const cv::Mat_<uchar>& intensity, dlib::frontal_face_detector& detector, std::vector<float>& o_confidences, float min_width, cv::Rect_<float> roi)
	{
		if (detector.num_detectors() == 0)
		{
			detector = dlib::get_frontal_face_detector();
		}

		cv::Mat_<uchar> upsampled_intensity;

		float scaling = 1.3f;

		cv::resize(intensity, upsampled_intensity, cv::Size((int)(intensity.cols * scaling), (int)(intensity.rows * scaling)));

		dlib::cv_image<uchar> cv_grayscale(upsampled_intensity);

		std::vector<dlib::full_detection> face_detections;
		detector(cv_grayscale, face_detections, -0.2);

		// Convert from int bounding box do a double one with corrections
		//o_regions.resize(face_detections.size());
		//o_confidences.resize(face_detections.size());

		for (size_t face = 0; face < face_detections.size(); ++face)
		{
			// CLNF expects the bounding box to encompass from eyebrow to chin in y, and from cheeck outline to cheeck outline in x, so we need to compensate

			cv::Rect_<float> region;
			// Move the face slightly to the right (as the width was made smaller)
			region.x = (face_detections[face].rect.get_rect().tl_corner().x() + 0.0389f * face_detections[face].rect.get_rect().width()) / scaling;
			// Shift face down as OpenCV Haar Cascade detects the forehead as well, and we're not interested
			region.y = (face_detections[face].rect.get_rect().tl_corner().y() + 0.1278f * face_detections[face].rect.get_rect().height()) / scaling;

			// Correct for scale
			region.width = (face_detections[face].rect.get_rect().width() * 0.9611) / scaling;
			region.height = (face_detections[face].rect.get_rect().height() * 0.9388) / scaling;

			// The scalings were learned using the Face Detections on LFPW and Helen using ground truth and detections from the HOG detector
			if (min_width != -1)
			{
				if (region.width < min_width || region.x < ((float)intensity.cols) * roi.x || region.y < ((float)intensity.cols) * roi.y ||
					region.x + region.width >((float)intensity.cols) * (roi.x + roi.width) || region.y + region.height >((float)intensity.rows) * (roi.y + roi.height))
					continue;
			}


			o_regions.push_back(region);
			o_confidences.push_back(face_detections[face].detection_confidence);


		}
		return o_regions.size() > 0;
	}

	bool DetectSingleFaceHOG(cv::Rect_<float>& o_region, const cv::Mat_<uchar>& intensity_img, dlib::frontal_face_detector& detector, float& confidence, cv::Point preference, float min_width, cv::Rect_<float> roi)
	{

		if (detector.num_detectors() == 0)
		{
			detector = dlib::get_frontal_face_detector();
		}

		// The tracker can return multiple faces
		vector<cv::Rect_<float> > face_detections;
		vector<float> confidences;
		bool detect_success = LandmarkDetector::DetectFacesHOG(face_detections, intensity_img, detector, confidences, min_width, roi);

		// In case of multiple faces pick the biggest one
		bool use_size = true;

		if (detect_success)
		{

			bool use_preferred = (preference.x != -1) && (preference.y != -1);

			// keep the most confident one or the one closest to preference point if set
			float best_so_far;
			if (use_preferred)
			{
				best_so_far = sqrt((preference.x - (face_detections[0].width / 2 + face_detections[0].x)) * (preference.x - (face_detections[0].width / 2 + face_detections[0].x)) +
					(preference.y - (face_detections[0].height / 2 + face_detections[0].y)) * (preference.y - (face_detections[0].height / 2 + face_detections[0].y)));
			}
			else if (use_size)
			{
				best_so_far = (face_detections[0].width + face_detections[0].height) / 2.0;
			}
			else
			{
				best_so_far = confidences[0];
			}
			int bestIndex = 0;

			for (size_t i = 1; i < face_detections.size(); ++i)
			{

				float dist;
				bool better;

				if (use_preferred)
				{
					dist = sqrt((preference.x - (face_detections[i].width / 2 + face_detections[i].x)) * (preference.x - (face_detections[i].width / 2 + face_detections[i].x)) +
						(preference.y - (face_detections[i].height / 2 + face_detections[i].y)) * (preference.y - (face_detections[i].height / 2 + face_detections[i].y)));

					better = dist < best_so_far;
				}
				else if (use_size)
				{
					dist = (face_detections[i].width + face_detections[i].height) / 2.0;
					better = dist > best_so_far;
				}
				else
				{
					dist = confidences[i];
					better = dist > best_so_far;
				}

				// Pick a closest face
				if (better)
				{
					best_so_far = dist;
					bestIndex = i;
				}
			}

			o_region = face_detections[bestIndex];
			confidence = confidences[bestIndex];
		}
		else
		{
			// if not detected
			o_region = cv::Rect_<float>(0, 0, 0, 0);
			// A completely unreliable detection (shouldn't really matter what is returned here)
			confidence = -2;
		}
		return detect_success;
	}

bool DetectFacesMTCNN(vector<cv::Rect_<float> >& o_regions, const cv::Mat& image, LandmarkDetector::FaceDetectorMTCNN& detector, std::vector<float>& o_confidences)
{
	detector.DetectFaces(o_regions, image, o_confidences);

	return o_regions.size() > 0;
}

bool DetectSingleFaceMTCNN(cv::Rect_<float>& o_region, const cv::Mat& image, LandmarkDetector::FaceDetectorMTCNN& detector, float& confidence, cv::Point preference)
{
	// The tracker can return multiple faces
	vector<cv::Rect_<float> > face_detections;
	vector<float> confidences;

	detector.DetectFaces(face_detections, image, confidences);

	bool detect_success = face_detections.size() > 0;
	if (detect_success)
	{

		bool use_preferred = (preference.x != -1) && (preference.y != -1);

		// keep the most confident one or the one closest to preference point if set
		float best_so_far;
		if (use_preferred)
		{
			best_so_far = sqrt((preference.x - (face_detections[0].width / 2 + face_detections[0].x)) * (preference.x - (face_detections[0].width / 2 + face_detections[0].x)) +
				(preference.y - (face_detections[0].height / 2 + face_detections[0].y)) * (preference.y - (face_detections[0].height / 2 + face_detections[0].y)));
		}
		else
		{
			best_so_far = face_detections[0].width;
		}
		int bestIndex = 0;

		for (size_t i = 1; i < face_detections.size(); ++i)
		{

			float dist;
			bool better;

			if (use_preferred)
			{
				dist = sqrt((preference.x - (face_detections[i].width / 2 + face_detections[i].x)) * (preference.x - (face_detections[i].width / 2 + face_detections[i].x)) +
					(preference.y - (face_detections[i].height / 2 + face_detections[i].y)) * (preference.y - (face_detections[i].height / 2 + face_detections[i].y)));
				better = dist < best_so_far;
			}
			else
			{
				dist = face_detections[i].width;
				better = dist > best_so_far;
			}

			// Pick a closest face
			if (better)
			{
				best_so_far = dist;
				bestIndex = i;
			}
		}

		o_region = face_detections[bestIndex];
		confidence = confidences[bestIndex];
	}
	else
	{
		// if not detected
		o_region = cv::Rect_<float>(0, 0, 0, 0);
		// A completely unreliable detection (shouldn't really matter what is returned here)
		confidence = -2;
	}
	return detect_success;
}


//============================================================================
// Matrix reading functionality
//============================================================================

// Reading in a matrix from a stream
void ReadMat(std::ifstream& stream, cv::Mat &output_mat)
{
	// Read in the number of rows, columns and the data type
	int row, col, type;

	stream >> row >> col >> type;

	output_mat = cv::Mat(row, col, type);

	switch (output_mat.type())
	{
	case CV_64FC1:
	{
		cv::MatIterator_<double> begin_it = output_mat.begin<double>();
		cv::MatIterator_<double> end_it = output_mat.end<double>();

		while (begin_it != end_it)
		{
			stream >> *begin_it++;
		}
	}
	break;
	case CV_32FC1:
	{
		cv::MatIterator_<float> begin_it = output_mat.begin<float>();
		cv::MatIterator_<float> end_it = output_mat.end<float>();

		while (begin_it != end_it)
		{
			stream >> *begin_it++;
		}
	}
	break;
	case CV_32SC1:
	{
		cv::MatIterator_<int> begin_it = output_mat.begin<int>();
		cv::MatIterator_<int> end_it = output_mat.end<int>();
		while (begin_it != end_it)
		{
			stream >> *begin_it++;
		}
	}
	break;
	case CV_8UC1:
	{
		cv::MatIterator_<uchar> begin_it = output_mat.begin<uchar>();
		cv::MatIterator_<uchar> end_it = output_mat.end<uchar>();
		while (begin_it != end_it)
		{
			stream >> *begin_it++;
		}
	}
	break;
	default:
		printf("ERROR(%s,%d) : Unsupported Matrix type %d!\n", __FILE__, __LINE__, output_mat.type()); abort();


	}
}

void ReadMatBin(std::ifstream& stream, cv::Mat &output_mat)
{
	// Read in the number of rows, columns and the data type
	int row, col, type;

	stream.read((char*)&row, 4);
	stream.read((char*)&col, 4);
	stream.read((char*)&type, 4);

	output_mat = cv::Mat(row, col, type);
	int size = output_mat.rows * output_mat.cols * output_mat.elemSize();
	stream.read((char *)output_mat.data, size);

}

// Skipping lines that start with # (together with empty lines)
void SkipComments(std::ifstream& stream)
{
	while (stream.peek() == '#' || stream.peek() == '\n' || stream.peek() == ' ' || stream.peek() == '\r')
	{
		std::string skipped;
		std::getline(stream, skipped);
	}
}

// Some other utility functions
void convert_to_grayscale(const cv::Mat& in, cv::Mat& out)
{
	if (in.channels() == 3)
	{
		// Make sure it's in a correct format
		if (in.depth() != CV_8U)
		{
			if (in.depth() == CV_16U)
			{
				cv::Mat tmp = in / 256;
				tmp.convertTo(tmp, CV_8U);
				cv::cvtColor(tmp, out, CV_BGR2GRAY);
			}
		}
		else
		{
			cv::cvtColor(in, out, CV_BGR2GRAY);
		}
	}
	else if (in.channels() == 4)
	{
		cv::cvtColor(in, out, CV_BGRA2GRAY);
	}
	else
	{
		if (in.depth() == CV_16U)
		{
			cv::Mat tmp = in / 256;
			out = tmp.clone();
		}
		else if (in.depth() != CV_8U)
		{
			in.convertTo(out, CV_8U);
		}
		else
		{
			out = in.clone();
		}
	}
}

void convert_to_8bit_bgr_or_grayscale(cv::Mat& in_out)
{
	if (in_out.channels() == 3)
	{
		// Make sure it's in a correct format
		if (in_out.depth() != CV_8U)
		{
			if (in_out.depth() == CV_16U)
			{
				in_out = in_out / 256;
				in_out.convertTo(in_out, CV_8UC3);
			}
			else if (in_out.depth() != CV_8U)
			{
				in_out.convertTo(in_out, CV_8U);
			}
		}
	}
	else if (in_out.channels() == 4)
	{
		cv::cvtColor(in_out, in_out, CV_BGRA2BGR);

		if (in_out.depth() == CV_16U)
		{
			in_out = in_out / 256;
			in_out.convertTo(in_out, CV_8UC3);
		}
	}
	else
	{
		if (in_out.depth() == CV_16U)
		{
			in_out = in_out / 256;
		}
		else if (in_out.depth() != CV_8U)
		{
			in_out.convertTo(in_out, CV_8U);
		}
	}
}

}
