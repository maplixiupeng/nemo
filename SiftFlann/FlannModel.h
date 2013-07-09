#pragma once
#include "nemo_dll/common.h"

namespace nemo
{
	struct FlannParameter 
	{
		string filename;
		string foldername;
		string corpusfoldername;
	};
class FlannModel
{
public:
	FlannModel(void){};
	~FlannModel(void){};

	void SiftFileFlannBuilder(string filename);

	void SiftFolderFlannBuilder(string foldername);

	void SiftCorpusFlannBuilder(string corpusFoldername);
};

}
