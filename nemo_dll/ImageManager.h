#pragma once
#include "common.h"
#include "FileManager.h"

namespace nemo
{

	static const cv::Vec3b bcolors[] =
	{
		cv::Vec3b(0,0,255),
		cv::Vec3b(0,128,255),
		cv::Vec3b(0,255,255),
		cv::Vec3b(0,255,0),
		cv::Vec3b(255,128,0),
		cv::Vec3b(255,255,0),
		cv::Vec3b(255,0,0),
		cv::Vec3b(255,0,255),
		cv::Vec3b(255,255,255)
	};

	class ImageManager
	{
	public:

		ImageManager(void){};

		~ImageManager(void){};

		static __declspec(dllexport)
			inline cv::Mat ImgSiftCollector(cv::Mat imat, bool isDisp = false, double sig = 1.6);

		//static __declspec(dllexport)
		//void ImgSiftCollectorFrFolder(vector<string> imgFolder, cv::Mat gps_mat);

		static __declspec(dllexport)
			inline cv::Mat ImgSurfCollector(cv::Mat imat, bool isDisp = false);

		static __declspec(dllexport)
			inline cv::Mat ImageClipper(cv::Mat imat);

		static __declspec(dllexport)
			void DrawSurfFeature(cv::Mat imat, vector<cv::KeyPoint> keypoints);

		static __declspec(dllexport)
			void FlannBuilder(CubeMat cm, string *flann_filename, cv::Mat mgps, string flann_folder);

		static __declspec(dllexport)
			void FlannBuilder(cv::Mat train_data, string flann_filename, string flann_folder);	

		static __declspec(dllexport)
			string FlannSearch(string img_filename, string flann_folder, cv::Mat imat);

		static __declspec(dllexport)
			string FlannSearch(string imgfolder, string flann_folder, int tree_idx=0, cv::Mat *gps_mat=0, bool isSift=true);

		static __declspec(dllexport)
			void MergeNNP(vector<NNP_Set> *orig, cv::Mat indices, cv::Mat dists, int *start_id, cv::Mat *gps_data);

		static __declspec(dllexport)
			void SaveFlannResult(string filename, vector<NNP_Set> nnps, int mat_width=0, int mat_heigh=0);

		static __declspec(dllexport)
			int MatlabMatGenerator(cv::Mat data_m, string filename);

		static __declspec(dllexport)
			cv::Mat DrawGPSHistogram(cv::Mat d_mat, string map_gps, float *alt, float *lng, int *max_voting=0);

		static __declspec(dllexport)
			cv::Mat MatchPruned(cv::Mat d_mat, int D);

		static __declspec(dllexport)
			cv::Mat BoWExtractor(cv::Mat d_mat, int centers = 1000);

		static __declspec(dllexport)
			void MSERFeatureDetector(cv::Mat imat);

		template<class T>
		static __declspec(dllexport)
			void PointProductBtwMats(cv::Mat imat1, cv::Mat imat2, cv::Mat *o_mat)
		{
			CV_Assert(imat1.type() == imat2.type());
			CV_Assert(imat1.size == imat2.size);

			if (!o_mat->data)
			{
				o_mat->create(imat1.rows, imat1.cols, imat1.type());
			}

			for (int i=0; i < imat1.rows; i++)
			{
				for (int j=0; j<imat1.cols; j++)
				{
					(*o_mat).at<T>(i, j) = imat1.at<T>(i, j) * imat2.at<T>(i, j);
				}
			}
		}

		static __declspec(dllexport)
			cv::Mat AffineTransform(cv::Mat inputMat);

		static __declspec(dllexport)
			cv::Mat AffineTransform2(cv::Mat inputMat);

		static __declspec(dllexport)
			cv::Mat RotationTransform(cv::Mat inputMat);

		static __declspec(dllexport)
			cv::Mat PanaImageClipper(cv::Mat imat);
	};

}
