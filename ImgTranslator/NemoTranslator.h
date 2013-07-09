#pragma once
#include "nemo_dll/common.h"

namespace nemo
{

class NemoTranslator
{
public:
	NemoTranslator(void){};
	~NemoTranslator(void){};

	void corpusTranslator(string corpusFolder, string wdFilename);

	void corpusToTrnDataFile(string docFolder, string trnFilename);

	
};

}
