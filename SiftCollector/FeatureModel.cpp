#include "FeatureModel.h"
#include "nemo_dll/FileManager.h"
#include "nemo_dll/ImageManager.h"

using namespace nemo;

/*!
	\fn FeatureCollector
Params:
	@param1: string imgFolder, e.g. img (which contain all sub image folders)
	@param2: string gpsFilename
	@param3: bool isSift

return 
	cv::Mat feature Mat

description
	Collect all feature (sift or surf) from google images. All images are clipped by bottom part(1/3)
	Feature mats are stored under "dataset" folder with name of "feature.sift", 
	the corresponding GPS mat is also stored under "dataset" folder with the name of "feature.gps"
*/
void FeatureModel::FeatureCollector(FeatureParams params, string imgFolder, string oFeatureFolder)
{
	FileManager fm;
	ImageManager im;
	vector<string> imgList = fm.FileReader(imgFolder, "\\*.jpg");
	CV_Assert(imgList.size()>0);

	//-----Prepare GPS mat for images
	cv::Mat mgps;
	fm.GpsDataReader(params.gpsFilename, &mgps);
	CV_Assert(mgps.data != 0);

	cv::Mat descriptors, descriptors_gps(0, 2, CV_64FC1);
	cv::Mat start_mat(0, 1, CV_32FC1); // Store start row index for each image
	int store_i=0;
	int img_Amt=1;
	for (int i=0; i<imgList.size(); i++)
	{
		string imgFilename = imgList[i];
		cv::Mat imat = cv::imread(imgFilename);
		imat = im.ImageClipper(imat);

		SiftGPS sg;
		sg.featureVector = im.ImgSiftCollector(imat, false, params.siftBlurCof);

		int imgIdx, viewIdx;
		fm.imageIdxExtractor(imgFilename, &imgIdx, &viewIdx);
		double lat = mgps.at<double>(imgIdx, 1);
		double lng = mgps.at<double>(imgIdx, 2);

		descriptors.push_back(sg.featureVector);
		cv::Mat _gps(sg.featureVector.rows, 2, CV_64FC1);
		_gps.colRange(0,1).setTo(lat);
		_gps.colRange(1,2).setTo(lng);

		descriptors_gps.push_back(_gps);

		start_mat.resize(img_Amt, cv::Scalar(descriptors.rows));
		img_Amt++;
		cout << "\rimage " << i << " is done...";
		double descriptorsSize = descriptors.rows * descriptors.cols * sizeof(float);
		if(descriptorsSize > NEMO_GB * params.fileSize)
		{
			//fm.DirectoryBuilder(params.outputFolder + "\\");
			char buffer[256];
			string _zeros = fm.zeros(4, store_i);
			sprintf(buffer, "%s\\%s%d.sift", oFeatureFolder.c_str(), _zeros.c_str(), store_i);
			fm.SaveMat2Disk<float>(descriptors, static_cast<string>(buffer));

			sprintf(buffer, "%s\\%s%d.gps", oFeatureFolder.c_str(), _zeros.c_str(), store_i);
			fm.SaveMat2Disk<double>(descriptors_gps, static_cast<string>(buffer));

			sprintf(buffer, "%s\\%s%d.sidx", oFeatureFolder.c_str(), _zeros.c_str(), store_i);
			fm.SaveMat2Disk<float>(start_mat, static_cast<string>(buffer));

			start_mat.release();
			descriptors.release();
			descriptors_gps.release();
			store_i++;
			img_Amt = 1;
		}
	}

	char buffer[256];
	//fm.DirectoryBuilder(oFeatureFolder + "\\");
	string zeros = fm.zeros(4, store_i);
	sprintf(buffer, "%s\\%s%d.sift", oFeatureFolder.c_str(), zeros.c_str(), store_i);			
	fm.SaveMat2Disk<float>(descriptors, static_cast<string>(buffer));

	sprintf(buffer, "%s\\%s%d.gps", oFeatureFolder.c_str(), zeros.c_str(), store_i);
	fm.SaveMat2Disk<double>(descriptors_gps, static_cast<string>(buffer));

	sprintf(buffer, "%s\\%s%d.sidx", oFeatureFolder.c_str(), zeros.c_str(), store_i);
	fm.SaveMat2Disk<float>(start_mat, static_cast<string>(buffer));
}

// Extract feature mat based on group folder
void FeatureModel::FeatureCollectorGp(FeatureParams params)
{
	FileManager fm;
	vector<string> folderList = fm.FolderReader(params.imgFolder);
	CV_Assert(folderList.size() > 0);
	
	for (int i=0; i<folderList.size(); i++)
	{
		vector<string> fl;
		fm.file_name_splitter(folderList[i], &fl, "\\");
		string foldername = params.outputFolder + "\\" + fl[fl.size()-1];
		foldername = fm.DirectoryBuilder(foldername + "\\");

		FeatureCollector(params, folderList[i], foldername);
	}	
}

// All images are organized into cubes and do sampling
void FeatureModel::FeatureCubeCollector(FeatureParams params)
{
	FileManager fm;
	ImageManager im;
	CubeMat cm = fm.CubeMatSampling(params.imgFolder, params.sampling);
	CV_Assert(cm.size() > 0);

	cv::Mat gps_mat;
	fm.GpsDataReader(params.gpsFilename, &gps_mat);
	CV_Assert(gps_mat.data != 0);

	vector<SiftGPS> vSG;
	cv::Mat descriptors;
	cv::Mat descriptors_gps(0,2,CV_64FC1);

	if(params.isAppend)
	{
		vector<string> vfilelist = fm.FileReader(params.outputFolder, "\\*.sift");

		if (vfilelist.size() > 0)
		{
			string siftfilename = vfilelist[vfilelist.size()-1];
			string gpsifilename = fm.FileExtensionChange(siftfilename, "gps");

			fm.ReadMatFromDisk<float>(siftfilename, &descriptors);
			fm.ReadMatFromDisk<double>(gpsifilename, &descriptors_gps);
		}		
	}

	int store_i=1;
	int cube_amt = 1;
	int img_Amt = 1;
	cv::Mat start_mat(0, 1, CV_32FC1);

	for (int i=0; i<cm.size(); i++)
	{
		vector<string> _cube = cm[i];
		for (int j=0; j < _cube.size(); j++)
		{
			string imgFilename = _cube[j];
			cv::Mat imat = cv::imread(imgFilename);
			imat = im.ImageClipper(imat);

			SiftGPS sg;
			if (params.featureType == "SIFT")
			{
				sg.featureVector = im.ImgSiftCollector(imat, false, params.siftBlurCof);
			}
			else
			{
				sg.featureVector = im.ImgSurfCollector(imat, false);
			}

			int imgIdx, viewIdx;
			fm.imageIdxExtractor(imgFilename, &imgIdx, &viewIdx);
			double lat = gps_mat.at<double>(imgIdx, 1);
			double lng = gps_mat.at<double>(imgIdx, 2);			

			descriptors.push_back(sg.featureVector);

			cv::Mat _gps(sg.featureVector.rows, 2, CV_64FC1);
			_gps.colRange(0, 1).setTo(lat);
			_gps.colRange(1, 2).setTo(lng);

			descriptors_gps.push_back(_gps);
			//cout << descriptors.row(descriptors.rows-1) << endl;
			//cout << descriptors_gps.row(descriptors_gps.rows-1) << endl;		
			start_mat.resize(img_Amt, cv::Scalar(descriptors.rows));
			img_Amt++;
		}

		
		//cout << start_mat.row(start_mat.rows-1) << endl;
		//start_mat.release();
		cout << "\rcube " << i << " is done." ;
		double descriptorSize = descriptors.rows * descriptors.cols * sizeof(float);
		if (descriptorSize > NEMO_GB*params.fileSize)
		{
			//Save descriptor mat into disk
			fm.DirectoryBuilder(params.outputFolder + "\\");

			char buffer[256];
			string zeros=fm.zeros(4, store_i);

			sprintf(buffer, "%s\\%s%d.sift", params.outputFolder.c_str(), zeros.c_str(), store_i);			
			fm.SaveMat2Disk<float>(descriptors, static_cast<string>(buffer));

			sprintf(buffer, "%s\\%s%d.gps", params.outputFolder.c_str(), zeros.c_str(), store_i);
			fm.SaveMat2Disk<double>(descriptors_gps, static_cast<string>(buffer));

			sprintf(buffer, "%s\\%s%d.sidx", params.outputFolder.c_str(), zeros.c_str(), store_i);
			fm.SaveMat2Disk<float>(start_mat, static_cast<string>(buffer));

			start_mat.release();
			descriptors.release();
			descriptors_gps.release();
			store_i++;
			cube_amt = 1;
			img_Amt = 1;
		}
	}

	//--------------------------------------------------

	fm.DirectoryBuilder(params.outputFolder + "\\");

	char buffer[256];
	string zeros= fm.zeros(4, store_i);

	sprintf(buffer, "%s\\%s%d.sift", params.outputFolder.c_str(), zeros.c_str(), store_i);			
	fm.SaveMat2Disk<float>(descriptors, static_cast<string>(buffer));

	sprintf(buffer, "%s\\%s%d.gps", params.outputFolder.c_str(), zeros.c_str(), store_i);
	fm.SaveMat2Disk<double>(descriptors_gps, static_cast<string>(buffer));

	sprintf(buffer, "%s\\%s%d.sidx", params.outputFolder.c_str(), zeros.c_str(), store_i);
	fm.SaveMat2Disk<float>(start_mat, static_cast<string>(buffer));
}