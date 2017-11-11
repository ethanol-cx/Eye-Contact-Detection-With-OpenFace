///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis, all rights reserved.
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

#include "VisualizationUtils.h"

using namespace Utilities;

void Project(cv::Mat_<double>& dest, const cv::Mat_<double>& mesh, double fx, double fy, double cx, double cy)
{
	dest = cv::Mat_<double>(mesh.rows, 2, 0.0);

	int num_points = mesh.rows;

	double X, Y, Z;


	cv::Mat_<double>::const_iterator mData = mesh.begin();
	cv::Mat_<double>::iterator projected = dest.begin();

	for (int i = 0; i < num_points; i++)
	{
		// Get the points
		X = *(mData++);
		Y = *(mData++);
		Z = *(mData++);

		double x;
		double y;

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

void DrawBox(cv::Mat image, cv::Vec6d pose, cv::Scalar color, int thickness, float fx, float fy, float cx, float cy)
{
	auto edge_lines = CalculateBox(pose, fx, fy, cx, cy);
	DrawBox(edge_lines, image, color, thickness);
}

std::vector<std::pair<cv::Point2d, cv::Point2d>> CalculateBox(cv::Vec6d pose, float fx, float fy, float cx, float cy)
{
	double boxVerts[] = { -1, 1, -1,
		1, 1, -1,
		1, 1, 1,
		-1, 1, 1,
		1, -1, 1,
		1, -1, -1,
		-1, -1, -1,
		-1, -1, 1 };

	std::vector<std::pair<int, int>> edges;
	edges.push_back(std::pair<int, int>(0, 1));
	edges.push_back(std::pair<int, int>(1, 2));
	edges.push_back(std::pair<int, int>(2, 3));
	edges.push_back(std::pair<int, int>(0, 3));
	edges.push_back(std::pair<int, int>(2, 4));
	edges.push_back(std::pair<int, int>(1, 5));
	edges.push_back(std::pair<int, int>(0, 6));
	edges.push_back(std::pair<int, int>(3, 7));
	edges.push_back(std::pair<int, int>(6, 5));
	edges.push_back(std::pair<int, int>(5, 4));
	edges.push_back(std::pair<int, int>(4, 7));
	edges.push_back(std::pair<int, int>(7, 6));

	// The size of the head is roughly 200mm x 200mm x 200mm
	cv::Mat_<double> box = cv::Mat(8, 3, CV_64F, boxVerts).clone() * 100;

	cv::Matx33d rot = LandmarkDetector::Euler2RotationMatrix(cv::Vec3d(pose[3], pose[4], pose[5]));
	cv::Mat_<double> rotBox;

	// Rotate the box
	rotBox = cv::Mat(rot) * box.t();
	rotBox = rotBox.t();

	// Move the bounding box to head position
	rotBox.col(0) = rotBox.col(0) + pose[0];
	rotBox.col(1) = rotBox.col(1) + pose[1];
	rotBox.col(2) = rotBox.col(2) + pose[2];

	// draw the lines
	cv::Mat_<double> rotBoxProj;
	Project(rotBoxProj, rotBox, fx, fy, cx, cy);

	std::vector<std::pair<cv::Point2d, cv::Point2d>> lines;

	for (size_t i = 0; i < edges.size(); ++i)
	{
		cv::Mat_<double> begin;
		cv::Mat_<double> end;

		rotBoxProj.row(edges[i].first).copyTo(begin);
		rotBoxProj.row(edges[i].second).copyTo(end);

		cv::Point2d p1(begin.at<double>(0), begin.at<double>(1));
		cv::Point2d p2(end.at<double>(0), end.at<double>(1));

		lines.push_back(std::pair<cv::Point2d, cv::Point2d>(p1, p2));

	}

	return lines;
}

void DrawBox(std::vector<std::pair<cv::Point, cv::Point>> lines, cv::Mat image, cv::Scalar color, int thickness)
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
