#pragma once
#include "nemo_dll\common.h"

namespace nemo
{

class FeatureModel
{
public:
	FeatureModel(void){};
	~FeatureModel(void){};

	void FeatureCollector(FeatureParams params, string imgFolder, string oFeatureFolder);

	void FeatureCollectorGp(FeatureParams params);

	void FeatureCubeCollector(FeatureParams params);
};

}
