#pragma once
#include "nemo_dll/common.h"

namespace nemo
{

class GroupModel
{
public:
	GroupModel(void){};
	~GroupModel(void){};

	void GroupingDBFrTheta(GroupParams gp);

        void FuzzyGroupDBFrTheta(GroupParams gp);

        void SoftGroup(cv::Mat cols, int col_idx, vector<vector<int>> *v_groups, double fuzzyThreshold);

        vector<string> ImageListGenerator(GroupParams gp);

        void CopyImg2Group(vector<vector<int>> v_group, string outputFolder, vector<string> imgList);
};

}
