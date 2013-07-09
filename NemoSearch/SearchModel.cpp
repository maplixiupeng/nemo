#include "SearchModel.h"
#include "nemo_dll/SearchManager.h"
#include "nemo_dll/FileManager.h"
#include "model.h"

using namespace nemo;

// 1. Entry program for Zheng's Search
void SearchModel::TestImgGrooup(SearchParams sp)
{
	string gtFilename = sp.testImgFolder + "\\gt.xml";
	bf::path p(gtFilename);
	if (!bf::exists(p))
	{
		std::cerr << "Fail to read ground true xml file under " << sp.testImgFolder << std::endl;
		exit;
	}

	double start_time = (double)cv::getTickCount();
	string testFilename = sp.testImgFolder + "\\test.xml";
	vector<testImgSift> vtis;
	bf::path p1(testFilename);
	if (bf::exists(p1))
	{
		std::cout << "Test files are ready, read backup files..." << std::endl;
		
		FileManager fm;
		vtis = fm.ReadTestImgDescriptor<float, uchar>(sp);
	}
	else
	{
		std::cout << "Without previous backup files, prepare test files..." << std::endl;
		SearchManager sm;
		vtis = sm.TestImagePrepare(sp);
		sm.TestImgTranslator(&vtis, sp);
		FileManager fm;
		fm.SaveTestImgDescriptor<float, uchar>(&vtis, sp);
	}
	
	
	start_time = ((double)cv::getTickCount() - start_time)/cv::getTickFrequency();
	std::cout << ">>Eclipse " << start_time << "s in test image preparation..." << std::endl;

	std::cout << "Save test image feature files under " << sp.testImgFolder << std::endl;
	
	std::cout << "Doing Topic inference..." << std::endl;
	string theta_filename = sp.modelFolder + "\\" + NEMO_TESTIMG_THETA_FILENAME;
	bf::path p_theta(theta_filename);
	if (!bf::exists(p_theta))
	{
		std::cout << "No theta files, start topic inference for test images..." << std::endl;
		start_time = (double)cv::getTickCount();
		TestImgInference(vtis, sp);
		start_time = ((double)cv::getTickCount() - start_time) / cv::getTickFrequency();
		std::cout << "Eclipse " << start_time << "s for test images inference..." <<std::endl;
	}	
	else
	{
		std::cout << "Find test theta file, start doing group..." << std::endl; 
	}

	std::cout << "Grouping test images based on topic..." << std::endl;
	start_time = (double)cv::getTickCount();
	GroupTestImg(sp, &vtis);
	start_time = ((double)cv::getTickCount() - start_time) / cv::getTickFrequency();
	std::cout << "Eclipse " << start_time << "s for image group..." << std::endl;

	std::cout << "Starting Flann Search in each group..." << std::endl;
	start_time = (double)cv::getTickCount();
	TestImgFlannSearch(vtis, sp);
	start_time = ((double)cv::getTickCount() - start_time) / cv::getTickFrequency();
	std::cout << "Eclipse " << start_time << "s for flann search..." << std::endl;
}

string SearchModel::TestImgInference(vector<testImgSift> imgList, SearchParams params)
{
	CV_Assert(imgList.size() > 0);
	vector<cv::Mat> vDocs;
	for (int i=0; i<imgList.size(); i++)
	{
		vDocs.push_back(imgList[i].words);
	}

	FileManager fm;
	string docFilename = params.modelFolder + "\\" + NEMO_TESTIMG_DATA_FILENAME;
	fm.saveImgClusterToTrnFile<float>(docFilename, vDocs);

	model lda;
	string model_name = params.modelName;//"model-final";
	string _dir = params.modelFolder;

	int argc = 11;
	char *argv[] = {"-inf", "-dir", const_cast<char*>(_dir.c_str()), "-model", const_cast<char*>(model_name.c_str()), "-niters", "30", "-twords", "20", "-dfile", NEMO_TESTIMG_DATA_FILENAME};

	if (lda.init(argc, argv))
	{
		cout << "Fail to prepare model data..." << endl;
	}

	if (lda.model_status == MODEL_STATUS_INF) {
		// do inference
		double start_time = cv::getTickFrequency();		
		lda.inference();
		start_time = (cv::getTickFrequency() - start_time) / cv::getTickCount();
		std::cout << "Eclipse " << start_time << " for test images inference..." << std::endl;
	}

	string thetaFilename = fm.FileExtensionChange(docFilename, "theta");
	return thetaFilename;
}

vector<testImgSift> SearchModel::GroupTestImg(SearchParams sp, vector<testImgSift> *vtis/* =0 */)
{
	FileManager fm;
	string testDataTheta = sp.modelFolder + "\\" + NEMO_TESTIMG_THETA_FILENAME;
	cv::Mat thetaMat;
	fm.ThetaDataReader(testDataTheta, &thetaMat);
	std::cout << "Read Theta file..." << std::endl;
	CV_Assert(thetaMat.data != 0);

	string centerFilename = sp.trGpFolder + "\\centers.mat";
	string centerFlannname = sp.trGpFolder + "\\centers.flann";

	cv::Mat center;
	fm.ReadMatFromDisk<float>(centerFilename, &center);
	CV_Assert(center.data != 0);
	cv::flann::Index index;
	index.load(center, centerFlannname);

	cv::Mat indics, distances;
	index.knnSearch(thetaMat, indics, distances, 1);
	indics.convertTo(indics, CV_32FC1);

	if (vtis == 0)
	{
		vtis = &(fm.ReadTestImgDescriptor<float, uchar>(sp));
	}

	for (int i=0; i <vtis->size(); i++)
	{
		(*vtis)[i].groups.push_back(indics.at<float>(i,0));
		std::cout << (*vtis)[i].imgname << "\t" << indics.at<float>(i,0) << std::endl;
	}

	//getchar();
	//std::cout << "Group finished, press any key to continue..." << std::endl;
	return *vtis;
}

void SearchModel::TestImgFlannSearch(vector<testImgSift> testList, SearchParams params)
{
	double st = (double)cv::getTickCount();

	FileManager fm;
	SearchManager sm;
	CV_Assert(testList.size() > 0);
	string featureFolder = params.trGpFolder + "\\features";
	vector<string> featureLists = fm.FolderReader(featureFolder);
	CV_Assert(featureLists.size() > 0);

	vector<vector<testImgSift>> gpTestList;
	gpTestList.resize(featureLists.size());	

	for (int i=0; i<testList.size(); i++)
	{
		testImgSift tis = testList[i];
		vector<float> _g = tis.groups;
		gpTestList[_g[0]].push_back(tis);
	}

	for (int i=0; i<gpTestList.size(); i++)
	{
		
		string _ffolder = featureLists[i];
		vector<string> flannList = fm.FileReader(_ffolder, "\\*.flann");
		CV_Assert(flannList.size());

		vector<testImgSift> testList = gpTestList[i]; // Test list

		if (testList.size() > 0)
		{
			double _st = (double)cv::getTickCount();
			string zeros = "0";
			int n = (i==0? 1:log10(i)+1);
			for (int z=0; z < 2-n; z++)
			{
				zeros = zeros + "0";
			}

			char buffer[256];
			std::sprintf(buffer, "%s\\%s%d", params.testImgFolder.c_str(), zeros.c_str(), i);
			string bkfolder = static_cast<string>(buffer);
			bf::path p_result(bkfolder+"\\result.txt");
			if (!bf::exists(p_result))
			{
				sm.FlannSearchFrImgList(testList, flannList, static_cast<string>(buffer));
			}
			
			sm.FlannSearchMerge(&testList, bkfolder, _ffolder);

			_st = ((double)cv::getTickCount() - _st) / cv::getTickFrequency();
			std::cout << "Flann Searching finish in group " << i << ", eclipse " << _st << "s..." << std::endl;
		}
		
	}

	//st = ((double)cv::getTickCount() - st)/cv::getTickFrequency();
	//std::cout << "Flann Search Finished in "<< st << "s, start merging...." << std::endl;
	//TestImgFlannSearchMerge(gpTestList, params);
}

void SearchModel::TestImgFlannSearchMerge(vector<vector<testImgSift>> gpTestList, SearchParams params)
{
	SearchManager sm;
	FileManager fm;
	string featureFolder = params.trGpFolder + "\\features";
	string bkupfolder = params.testImgFolder;
	vector<string> testFolder = fm.FolderReader(bkupfolder);
	vector<string> gpsFolder = fm.FolderReader(featureFolder);

	for (int i=0; i<testFolder.size(); i++)
	{
		std::cout << "\rDeal with " << i << " test folder";
		string _tf = testFolder[i];
		string resultFilename = _tf + "\\result.txt";
		bf::path p(resultFilename);
		if (!bf::exists(p))
		{
			vector<string> _vtf;
			fm.file_name_splitter(_tf, &_vtf, "\\");
			string _fidx = _vtf[_vtf.size() - 1];
			int fidx = atoi(_fidx.c_str());
			string _gpsFolder = gpsFolder[fidx];
			vector<testImgSift> testList = gpTestList[fidx];

			sm.FlannSearchMerge(&testList, _tf, _gpsFolder);
			gpTestList[fidx] = testList;
		}		
	}
}

// 2. Entry program for Amir's work

void SearchModel::TestAmirPlainSearch(SearchParams sp)
{
	SearchManager sm;
	FileManager fm;
    vector<testImgSift> vtis;
	string testFilename = sp.testImgFolder + "\\test.xml";
	bf::path p(testFilename);
	if(bf::exists(p))
	{
		vtis = fm.ReadTestImgDescriptor<float, uchar>(sp);
	}
	else
	{
		vtis = sm.TestImagePrepare(sp);
		fm.SaveTestImgDescriptor<float, uchar>(&vtis, sp);
	}
	CV_Assert(vtis.size() > 0);

	double start_time  = (double)cv::getTickCount();
	sm.AmirFlannSearch(&vtis, sp);
	start_time = ((double)cv::getTickCount() - start_time)/cv::getTickFrequency();
	std::cout << "Eclipse " << start_time << "s in Searching..." << std::endl;

	start_time = (double)cv::getTickCount();
	sm.AmirFlannSearchMerge(&vtis, sp);
	start_time = ((double)cv::getTickCount() - start_time) / cv::getTickFrequency();
	std::cout << "Eclipse " << start_time << "s in merging..." << std::endl;
}

// 3. Entry program for zheng's multi groups searching
void SearchModel::TestZhengSceneSearch(SearchParams sp)
{
	string gtFilename = sp.testImgFolder + "\\gt.xml";
	bf::path p(gtFilename);
	if (!bf::exists(p))
	{
		std::cerr << "Fail to read ground true xml file under " << sp.testImgFolder << std::endl;
		exit;
	}
	FileManager fm;

	double start_time = (double)cv::getTickCount();
	string testFilename = sp.testImgFolder + "\\test.xml";
	vector<testImgSift> vtis;
	bf::path p1(testFilename);
	if (bf::exists(p1))
	{
		std::cout << "Test files are ready, read backup files..." << std::endl;
		
		vtis = fm.ReadTestImgDescriptor<float, uchar>(sp);
	}
	else
	{
		std::cout << "Without previous backup files, prepare test files..." << std::endl;
		SearchManager sm;
		vtis = sm.TestImagePrepare(sp);
		sm.TestImgTranslator(&vtis, sp);
		fm.SaveTestImgDescriptor<float, uchar>(&vtis, sp);
	}

	start_time = ((double)cv::getTickCount() - start_time)/cv::getTickFrequency();
	std::cout << ">>Eclipse " << start_time << "s in test image preparation..." << std::endl;

	std::cout << "Save test image feature files under " << sp.testImgFolder << std::endl;

	std::cout << "Doing Topic inference..." << std::endl;
	string theta_filename = sp.modelFolder + "\\" + NEMO_TESTIMG_THETA_FILENAME;
	bf::path p_theta(theta_filename);
	if (!bf::exists(p_theta))
	{
		std::cout << "No theta files, start topic inference for test images..." << std::endl;
		start_time = (double)cv::getTickCount();
		TestImgInference(vtis, sp);
		start_time = ((double)cv::getTickCount() - start_time) / cv::getTickFrequency();
		std::cout << "Eclipse " << start_time << "s for test images inference..." <<std::endl;
	}	
	else
	{
		std::cout << "Find test theta file, start doing group..." << std::endl; 
	}

	std::cout << "Grouping test images based topic..." << std::endl;
	start_time = (double)cv::getTickCount();
	vector<vector<testImgSift>> v_groups = GroupTestImgMultipleGPs(sp, &vtis);
	start_time = ((double)cv::getTickCount() - start_time) / cv::getTickFrequency();
	std::cout << "Eclipse " << start_time << "s for image group..." << std::endl;

	SearchManager sm;
	sm.FlannSearchMultipleGP(v_groups, sp);
}

vector<vector<testImgSift>> SearchModel::GroupTestImgMultipleGPs(SearchParams sp, vector<testImgSift> *vtis)
{
	CV_Assert(vtis->size() > 0);
	FileManager fm;
	string testDataTheta = sp.modelFolder + "\\" + NEMO_TESTIMG_THETA_FILENAME;
	cv::Mat thetaMat;
	fm.ThetaDataReader(testDataTheta, &thetaMat);
	std::cout << "Read Theta file...." << std::endl;
	CV_Assert(thetaMat.data != 0);

	string centerFilename = sp.trGpFolder + "\\centers.mat";
	string centerFlannname = sp.trGpFolder + "\\centers.flann";

	cv::Mat center;
	fm.ReadMatFromDisk<float>(centerFilename, &center);
	CV_Assert(center.data != 0);
	cv::flann::Index index;
	index.load(center, centerFlannname);

	cv::Mat indics, distances;
	index.knnSearch(thetaMat, indics, distances, 5);
	indics.convertTo(indics, CV_32FC1);

	for (int i=0; i<vtis->size(); i++)
	{
		cv::Mat gp_row = indics.row(i);
		cv::Mat gp_dist_row = distances.row(i);

		float min_dist = gp_dist_row.at<float>(0, 0);
		(*vtis)[i].groups.push_back(indics.at<float>(i, 0));

		for (int j=1; j<gp_row.cols; j++)
		{
			float _dist = gp_dist_row.at<float>(0, j);
			if (min_dist / _dist >= 0.45)
			{
				(*vtis)[i].groups.push_back(indics.at<float>(i, j));
				std::cout << (*vtis)[i].imgname << "\t" << indics.at<float>(i, j) << std::endl;
			}			
		}		
	}

	vector<vector<testImgSift>> v_groups(center.rows);
	for (int i=0; i<indics.rows; i++)
	{
		cv::Mat gp_row = indics.row(i);
		cv::Mat gp_dist_row = distances.row(i);

		float min_dist = gp_dist_row.at<float>(0, 0);
		float min_gp = gp_row.at<float>(0, 0);
		v_groups[min_gp].push_back((*vtis)[i]);

		for (int j=1; j<gp_row.cols; j++)
		{
			float _dist = gp_dist_row.at<float>(0, j);
			if (min_dist/_dist >=0.9)
			{
				float gp_id = gp_row.at<float>(i, j);
				v_groups[gp_id].push_back((*vtis)[i]);
			}			
		}
	}
	return v_groups;
}