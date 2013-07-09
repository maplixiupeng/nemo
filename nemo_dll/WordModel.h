#pragma once
#include "common.h"

namespace nemo
{

	class WordModel
	{
	public:
		WordModel(void){};
		~WordModel(void){};

		static __declspec(dllexport)
			cv::Mat generateWordDictionary(cv::Mat allFeatureSamples, int numberOfClusters);

		static __declspec(dllexport)
			cv::flann::Index generateFlannIndex(cv::Mat clusterSamples, int km_branches=32, int km_it=11);

		static __declspec(dllexport)
			cv::Mat generateClusterMembershipForFeatures(cv::flann::Index flann_index, cv::Mat featureVector, int k=1);

		template <class T>
		static __declspec(dllexport)    
			void saveImgClusterToTrnFile(string filename, vector<cv::Mat> documents);

		static __declspec(dllexport)    
			vector<cv::Mat> imgTranslator(string imgFolder, string docFilename, string wordFolder="");

		static __declspec(dllexport)   
			cv::Mat clusterImageBasedonTopicDist(string thetaFilename, string imgfolder);
	private:
		int numberOfClusters;

	};

}