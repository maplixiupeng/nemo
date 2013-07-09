#pragma once
#include "nemo_dll/common.h"

namespace nemo
{
	
class SearchModel
{
public:
	SearchModel(void){};
	~SearchModel(void){};

	void TestImgGrooup(SearchParams sp);

	string TestImgInference(vector<testImgSift> imgList, SearchParams params);

	vector<testImgSift> GroupTestImg(SearchParams sp, vector<testImgSift> *vtis=0);

	vector<vector<testImgSift>> GroupTestImgMultipleGPs(SearchParams sp, vector<testImgSift> *vtis);

	void TestImgFlannSearch(vector<testImgSift> testList, SearchParams params);

	void TestImgFlannSearchMerge(vector<vector<testImgSift>> gpTestList, SearchParams params);

	void TestAmirPlainSearch(SearchParams sp);

	void TestZhengSceneSearch(SearchParams sp);
};

}
