#include "NemoTranslator.h"
#include "nemo_dll/FileManager.h"

using namespace nemo;

// Input folder contain all feature mats, gps mats and sidx files
void NemoTranslator::corpusTranslator(string corpusFolder, string wdFilename)
{
	FileManager fm;
	vector<string> matList = fm.FileReader(corpusFolder, "\\*.sift");
	CV_Assert(matList.size() > 0);

	std::cout << "Load word index..." << std::endl;
	bf::path p(wdFilename);
	CV_Assert(bf::exists(p));
	string wdFlannfilename = fm.FileExtensionChange(wdFilename, "flann");
	bf::path pf(wdFlannfilename);
	CV_Assert(bf::exists(pf));
	cv::Mat mwd;
	fm.ReadMatFromDisk<float>(wdFilename, &mwd);
	cv::flann::Index widx;
	widx.load(mwd, wdFlannfilename);

	std::cout << "Start image translator for " << matList.size() << std::endl;
	for (int i=0; i<matList.size(); i++)
	{
		string matFilename = matList[i];
		string sidxFilename = fm.FileExtensionChange(matFilename, "sidx");
		bf::path p(sidxFilename);
		if (!bf::exists(p))
		{
			std::cerr << "Cannot find sidx file for" << matFilename << std::endl;
			exit;
		}
		string docsFilename = fm.FileExtensionChange(matFilename, "doc");
		bf::path pdoc(docsFilename);
		if (!bf::exists(pdoc))
		{		
			cv::Mat msidx, msift;
			fm.ReadMatFromDisk<float>(matFilename, &msift);
			fm.ReadMatFromDisk<float>(sidxFilename, &msidx);

			CV_Assert(msift.data != 0);
			CV_Assert(msidx.data != 0);

			float start_i=0;
			vector<cv::Mat> vDocs;
		
			for (int j=0; j < msidx.rows; j++)
			{
				float _idx = msidx.at<float>(0, j);
				cv::Mat img = msift.rowRange(start_i, _idx);
				start_i = _idx;

				cv::Mat indics, dist;
				widx.knnSearch(img, indics, dist, 1);
				indics.convertTo(indics, CV_32FC1);
				vDocs.push_back(indics);
			}
			fm.SaveMatVect2Disk<float>(vDocs, docsFilename);
		}	
		
		std::cout << "\rMat_" << i << std::endl;
	}
}

void NemoTranslator::corpusToTrnDataFile(string docFolder, string trnFilename)
{
	FileManager fm;
	vector<string> docList;
	docList = fm.FileReader(docFolder, "\\*.doc");
	CV_Assert(docList.size() > 0);

	bf::path p(trnFilename);
	if (!bf::exists(p.parent_path()))
	{
		std::cout << p.parent_path().string() << " doesn't existed,  create directory..." << std::endl;
		fm.DirectoryBuilder(p.parent_path().string() + "\\");
	}

	vector<cv::Mat> vDocs;
	for (int i=0; i<docList.size(); i++)
	{
		vector<cv::Mat> _vd;
		fm.ReadMatVect2Disk<float>(docList[i], &_vd);
		CV_Assert(_vd.size() > 0);
		
		for (int j=0; j<_vd.size(); j++)
		{
			vDocs.push_back(_vd[j]);
		}
	}
	std::cout << "Save corpus data..." << std::endl;
	string corpusFilename = docFolder + "\\corpus.doc";
	fm.SaveMatVect2Disk<float>(vDocs, corpusFilename);

	std::cout << "Translate to trn format..." << std::endl;
	
	
	fm.saveImgClusterToTrnFile<float>(trnFilename, vDocs);
}