#pragma once
#include "nemo_dll/common.h"
#include "nemo_dll/FileManager.h"

namespace nemo
{

    class NemoUtilities
    {
    public:
        NemoUtilities(void){};
        ~NemoUtilities(void){};

        void wordGenerate(string featureFolder, double wdAmt);

        void corpus2WDList(string corpusFolder, string wdFilename);

        void corpus2TrnDatafile(string docFolder, string trnFilename);

        void MatlabMatGenerator(string thetaFilename, string oMatfilename="");

		void DrawVisualWord(string imgListFolder, string wdfolder);

		cv::Mat DrawColorSifts(cv::Mat indics, cv::Mat image, vector<cv::KeyPoint> keypoints);


    };

}
