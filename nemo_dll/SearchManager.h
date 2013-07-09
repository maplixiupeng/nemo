#pragma once
#include "common.h"

namespace nemo
{

	class SearchManager
	{
	public:
		SearchManager(void){};
		~SearchManager(void){};

		// 1. Test image prepare: 
		// Extract Feature points
		static __declspec(dllexport) vector<testImgSift> TestImagePrepare(SearchParams params);

		static __declspec(dllexport) void TestImgTranslator(vector<testImgSift> *imgList, SearchParams params);		

		static __declspec(dllexport) void FeatureFlannBuilder(vector<string> featureFolders);

		static __declspec(dllexport) void FlannSearchFrImgList(vector<testImgSift> imgList, vector<string> flannList, string bkupFolder);

		static __declspec(dllexport) void FlannSearchMerge(vector<testImgSift> *imgList, string bkupFolder, string featureFolder);

		static void Voting(vector<testImgSift> *vtis, SearchParams params);

		static void FeaturePruned(vector<testImgSift> *vtis, SearchParams sp);

		static void printNNP_Set(NNP_Set inNNPs)
		{
			const NNP_Set::nth_index<1>::type &dist = inNNPs.get<1>();
			bm::nth_index<NNP_Set, 1>::type::iterator it = dist.begin();

			for (; it != get<1>(inNNPs).end(); it++)
			{
				cout << it->indics << "\t" << it->dist << "\t" << it->alt << "\t" << it->lng << "\n";
			}
			cout << "\n";
		}

		static double GPSDistance(double lat1, double lng1, double lat2, double lng2)
		{
			double distance = sqrt(pow((lat1 - lat2), 2) + pow((lng1 - lng2), 2));
			return distance;
		}

		// ------- The following two methods are working for Amir's framework
		static __declspec(dllexport) void AmirFlannSearch(vector<testImgSift> *vtis, SearchParams params);

		static __declspec(dllexport) void AmirFlannSearchMerge(vector<testImgSift> *vtis, SearchParams params);

		static __declspec(dllexport) void FlannSearchMultipleGP(vector<vector<testImgSift>> v_groups, SearchParams params);

		static __declspec(dllexport) void FlannSearchMultipleGPPer(vector<testImgSift> vtis, vector<string> flannList, int scene_id, int flann_id, int img_id, string bkFolder);

		static void Preresult(string bkFolder, int *scene_id, int *flann_id, int *img_id);
	};

}
