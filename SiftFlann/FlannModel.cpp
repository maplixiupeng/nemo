#include "FlannModel.h"
#include "nemo_dll/FileManager.h"
#include "nemo_dll/WordModel.h"

using namespace nemo;

// Building Flann index for single file
void FlannModel::SiftFileFlannBuilder(string filename)
{
	FileManager fm;
	bf::path p(filename);
	if (!bf::exists(p))
	{
		std::cerr << filename << " doesn't exist..." << std::endl;
		return;
	}

	WordModel wm;
	cv::Mat mData;
	fm.ReadMatFromDisk<float>(filename, &mData);
	CV_Assert(mData.data != 0);
	cv::flann::Index idx = wm.generateFlannIndex(mData);
	string idxFilename = fm.FileExtensionChange(filename, "flann");
	idx.save(idxFilename);
}

// Building flann index for sift files under same folder
void FlannModel::SiftFolderFlannBuilder(string foldername)
{
	FileManager fm;
	vector<string> fileList = fm.FileReader(foldername, "\\*.sift");
	CV_Assert(fileList.size() > 0);

	std::cout << "Doing Flann " << fileList.size() << std::endl;

	for (int i=0; i<fileList.size(); i++)
	{
		string flannFilename = fm.FileExtensionChange(fileList[i], "flann");
		bf::path p(flannFilename);
		if (!bf::exists(p))
		{
			SiftFileFlannBuilder(fileList[i]);
		}
		std::cout << "\rFlann " << i ;
	}
}

// Building Flann index for sift folder under given folder
void FlannModel::SiftCorpusFlannBuilder(string corpusFoldername)
{ 
	FileManager fm;
	vector<string> folderList = fm.FolderReader(corpusFoldername);
	CV_Assert(folderList.size() > 0);
	
	std::cout << "Doing flann on " << folderList.size() << " folders...\n";

	for (int i=0; i < folderList.size(); i++)
	{
		string foldername = folderList[i];
		SiftFolderFlannBuilder(foldername);
		std::cout << "\rFolder " << i << " is finished...";
	}
}