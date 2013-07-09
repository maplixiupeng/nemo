#include "SearchManager.h"
#include "FileManager.h"
#include "ImageManager.h"
#include "WordModel.h"

using namespace nemo;

// Construct flann index tree for all feature folders
void SearchManager::FeatureFlannBuilder(vector<string> featureFolders)
{
	FileManager fm;
	WordModel wm;
	CV_Assert(featureFolders.size() > 0);
	std::cout << "Constructing Flann index for " << featureFolders.size() << " folders" << std::endl;
	for (int i = 0; i < featureFolders.size(); i++)
	{
		vector<string> siftList = fm.FileReader(featureFolders[i], "\\*.sift");
		std::cout << "\rConstruct Folder " << i;
		for (int j=0; j< siftList.size(); j++)
		{
			string flannFilename = fm.FileExtensionChange(siftList[j], "flann");
			bf::path p(flannFilename);
			if(!bf::exists(p))
			{
				cv::Mat sift;
				fm.ReadMatFromDisk<float>(siftList[j], &sift);
				CV_Assert(sift.data != 0);						
				cv::flann::Index idx(sift, cv::flann::KMeansIndexParams());
#if _DEBUG
				cv::Mat indics, dist;
				idx.knnSearch(sift, indics, dist, 1);
#endif
				idx.save(flannFilename);
			}
		}
	}
}

void SearchManager::FlannSearchFrImgList(vector<testImgSift> imgList, vector<string> flannList, string bkupFolder)
{
	CV_Assert(imgList.size() > 0);
	CV_Assert(flannList.size() > 0);
	FileManager fm;
	fm.DirectoryBuilder(bkupFolder);
	bf::path p(bkupFolder);
	if (!bf::exists(p))
	{
		bkupFolder = bkupFolder + "\\";
		fm.DirectoryBuilder(bkupFolder);
	}

	std::cout << " \n Start Doing Flann Searching..." << std::endl;
	int j, i;
	vector<string> pre_result = fm.FileReader(bkupFolder, "\\*.indics");
	if (pre_result.size() == 0)
	{
		std::cout << "Without previous result, start from the first one..." << std::endl;
		j=0;
		i = 0;
	}
	else
	{
		string last_result = pre_result[pre_result.size()-1];
		bf::path p(last_result);
		last_result = p.filename().string();
		vector<string> fmList;
		fm.file_name_splitter(last_result, &fmList, "_");

		string j_str = fmList[fmList.size()-3]; // Image index number
		string i_str = fmList[fmList.size()-2];
		j = atoi(j_str.c_str()) + 1;
		if (j == imgList.size())
		{
			//All images are finished in previous flann index
			i = atoi(i_str.c_str()) + 1;
			j = 0;
		}
		else
		{
			// Previous flann search is stopped at img j
			i = atoi(i_str.c_str());
		}
	}

	std::cout << "Start from flann_" << i << std::endl;

	for (; i < flannList.size(); i++)
	{
		string flannFilename = flannList[i];
		string featureFilename = fm.FileExtensionChange(flannList[i], "sift");
		string gpsFilename = fm.FileExtensionChange(flannList[i], "gps");

		cv::Mat sift, gps;
		fm.ReadMatFromDisk<float>(featureFilename, &sift);
		fm.ReadMatFromDisk<double>(gpsFilename, &gps);
		CV_Assert(sift.data != 0);
		CV_Assert(gps.data != 0);

		cv::flann::Index index;
		index.load(sift, flannFilename);

		string zeros = "0";
		int n = (i==0? 1:log10(i)+1);
		for (int z=0; z < 4-n; z++)
		{
			zeros = zeros + "0";
		}

		for (; j<imgList.size(); j++)
		{
			testImgSift tis = imgList[j];
			bf::path p(tis.imgname);
			string imgfilename = p.filename().string();
			char buffer_indics[256], buffer_dist[256];

			string _zeros = "0";
			int _n = (j==0? 1: log10(j) + 1);
			for (int z = 0; z < 4 - _n; z++)
			{
				_zeros = _zeros + "0";
			}

			sprintf(buffer_indics, "%s\\%s%d_%s%d_%s.indics", bkupFolder.c_str(), _zeros.c_str(), j,  zeros.c_str(), i, imgfilename.c_str());
			sprintf(buffer_dist, "%s\\%s%d_%s%d_%s.dist", bkupFolder.c_str(), _zeros.c_str(), j, zeros.c_str(), i, imgfilename.c_str());
			string logFilename = bkupFolder + "\\time.log";

			cv::Mat indics, dists;
			double st = (double)cv::getTickCount();
			index.knnSearch(tis.descriptor, indics, dists, 5);
			st = ((double)cv::getTickCount() - st) / cv::getTickFrequency();
			string msg = ">>>>>>>>>>>>>>>>>>>>>>Flann Searching Time<<<<<<<<<<<<<<<<<<<<<<<\n";
			std::cout << msg;
			//fm.WriteTimeLog(msg, logFilename);
			
			std::cout << "Eclipse " << st << "s" <<  std::endl;
			char log_buffer[256];
			sprintf(log_buffer, "%lf\n", st);
			fm.WriteTimeLog(static_cast<string>(log_buffer), logFilename);

			msg = ">>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
			std::cout << msg;
			//fm.WriteTimeLog(msg, logFilename);

			indics.convertTo(indics, CV_32FC1);
			fm.SaveMat2Disk<float>(indics, static_cast<string>(buffer_indics));
			fm.SaveMat2Disk<float>(dists, static_cast<string>(buffer_dist));
		}
		j=0;
		std::cout << "\rFlann_" << i << " is done...";
	}
}

// featureFolder: group feature folder contains gps file and sift file
void SearchManager::FlannSearchMerge(vector<testImgSift> *imgList, string bkupFolder, string featureFolder)
{
	FileManager fm;
	CV_Assert(imgList->size() > 0);
	vector<string> distList = fm.FileReader(bkupFolder, "\\*.dist");
	vector<string> indicsList = fm.FileReader(bkupFolder, "\\*.indics");
	CV_Assert(distList.size() > 0);
	CV_Assert(distList.size() == indicsList.size());

	// Get how many flann index used
	vector<string> gpsList = fm.FileReader(featureFolder, "\\*.gps");
	int flann_amt = gpsList.size();

	// 1. Prepare GPS mat data
	std::cout << "Prepare GPS vector..." << std::endl;
	vector<cv::Mat> vGps;
	for (int i=0; i<gpsList.size(); i++)
	{
		cv::Mat _gps;
		fm.ReadMatFromDisk<double>(gpsList[i], &_gps);
		vGps.push_back(_gps);
	}

	// 2. Prepare all distances and indics mat
	std::cout << "Prepare all distances and indics mat..." << std::endl;
	vector<cv::Mat> vDists, vIndics;
	for (int i=0; i<distList.size(); i++)
	{
		cv::Mat dist, indics;
		fm.ReadMatFromDisk<float>(distList[i], &dist);
		fm.ReadMatFromDisk<float>(indicsList[i], &indics);
		vDists.push_back(dist);
		vIndics.push_back(indics);
	}

	// 3. Merge searching results
	std::cout << "Merging searching result..." << endl;
	int _i=0;
	for(int j=0; j<imgList->size(); j++) // For each image
	{
		testImgSift tis = (*imgList)[j];
		tis.vnnps.resize(tis.descriptor.rows);

		for (; _i<(j+1)*flann_amt; _i++)
		{
			cv::Mat dist, indics;
			dist = vDists[_i];
			indics = vIndics[_i];

			int flann_id = _i%flann_amt;
			cv::Mat _gps = vGps[flann_id];

			for (int k=0; k<dist.rows; k++)
			{
				for (int d=0; d<dist.cols; d++)
				{
					int id = flann_id * 5 + d;
					double _indics = indics.at<float>(k, d);
					double _dist = dist.at<float>(k, d);
					double _lat = _gps.at<double>(_indics, 0);
					double _lng = _gps.at<double>(_indics, 1);
					NNPoint nnp(id, _indics, _dist, _lat, _lng);
					tis.vnnps[k].insert(nnp);
				}
			}
		}
		(*imgList)[j] = tis;
		std::cout << "img_"<<j<<" is merged..." << endl;
	}
	SearchParams params;
	params.outputFolder = bkupFolder;
	params.accuracy_dist = 0.002;
	Voting(imgList, params);
}

void SearchManager::Voting(vector<testImgSift> *vtis, SearchParams params)
{
	if(true)
		FeaturePruned(vtis, params);

	for (int i=0; i<(*vtis).size(); i++)
	{
		testImgSift tis = (*vtis)[i];
		tis.voting_amt = 0;
		vector<NNP_Set> vnnps = tis.vnnps;
		GPS_Set gs;
		GPS_by_latlng::iterator gs_it;

		for (int j=0; j < vnnps.size(); j++) // for one image
		{
			if (tis.voting_tag.at<uchar>(j,0) != 0) // for one feature point
			{
				NNP_Set nnp = vnnps[j];
				const NNP_Set::nth_index<1>::type &dist = nnp.get<1>();
				bm::nth_index<NNP_Set, 1>::type::iterator it = dist.begin();

				{
					gs_it = gs.find(boost::make_tuple(it->alt, it->lng));
					if (gs_it == gs.end()) // new GPS location
					{
						gs.insert(GPS_Tuple(j, it->alt, it->lng, 1));
					}
					else
					{
						GPS_Tuple gt = *gs_it;
						gt.voting++;
						gs.replace(gs_it, gt);
					}
				}
				tis.voting_amt++;
			}
		}
		tis.gs = gs;
		(*vtis)[i] = tis;
	}

	//---------Find the maximum voting--------------//
	ofstream resultFile(params.outputFolder + "\\result.txt");
	int count = 0;
	for (int i =0; i < (*vtis).size(); i++)
	{
		testImgSift tis = (*vtis)[i];
		GPS_by_latlng::iterator it = tis.gs.begin();
		int max_voting = INT_MIN;
		double lat, lng;

		for (; it != tis.gs.end(); it++)
		{
			if(it->voting > max_voting)
			{
				max_voting = it->voting;
				lat = it->lat;
				lng = it->lng;
			}
		}

		tis.lat = lat;
		tis.lng = lng;
		double distance = sqrt(pow((tis.glat - lat), 2) + pow((tis.glng - lng), 2));
		float amt = tis.voting_amt;
#if _DEBUG
		cout << "Good Voting number.." << amt << endl;
		cout << "Total FP.. " << tis.words.rows << endl;
#endif
		if (distance < params.accuracy_dist)
		{
			count++;
		}
		if (resultFile.is_open())
		{
			char buffer[256];
			std::sprintf(buffer, "img:%s, lat: %lf, lng: %lf, dist: %lf, goodvoting: %d, totalvoting: %d, voting: %d, conf: %lf\n", tis.imgname.c_str(), lat, lng, distance, tis.voting_amt, tis.descriptor.rows, max_voting, (double)max_voting/amt);
			resultFile << static_cast<string>(buffer);
		}
		printf("\nimg:%s, lat: %lf, lng: %lf, voting: %d, dist: %lf, conf: %lf\n", tis.imgname.c_str(), lat, lng, max_voting, distance, (double)max_voting/amt);		
	}

	char buffer[256];
	std::sprintf(buffer, "correct: %d\n", count);
	resultFile << static_cast<string>(buffer);
	resultFile.close();
}

void SearchManager::FeaturePruned(vector<testImgSift> *vtis, SearchParams sp)
{
	CV_Assert(vtis->size() > 0);

	std::cout << "Doing Pruning..." << std::endl;

	for (int i = 0; i < vtis->size(); i++)
	{

		testImgSift tis = (*vtis)[i];		

		std::cout << "Pruning img_" << i << " with " << tis.vnnps.size() << " feature points." << endl;

		vector<NNP_Set> vnnps = tis.vnnps;
		vector<NNP_Set>::iterator it = vnnps.begin();

		for (int k =0; k < tis.vnnps.size(); k++) //--For one image
		{			
			NNP_Set nns = tis.vnnps[k];
			const NNP_Set::nth_index<1>::type &dist = nns.get<1>();
			bm::nth_index<NNP_Set, 1>::type::iterator _it = dist.begin();
			double ddist_1 = _it->dist;
			double nn_1_lat = _it->alt;
			double nn_1_lng = _it->lng;

			int j = INT_MAX;
			_it++;
			int nn=1;			
			double ddist_j = 0;
			for (; _it != get<1>(nns).end(); _it++) //----For one Feature Point
			{
				double nn_it_lat = _it->alt;
				double nn_it_lng = _it->lng;
				double dist = GPSDistance(nn_1_lat, nn_1_lng, nn_it_lat, nn_it_lng);

				if (dist > 0.0006 && j > nn) // prunD
				{
					j = nn; //record Min(j)
					ddist_j = _it->dist;
				}
			}

			if (ddist_j != 0)
			{
				double f_dist = ddist_1 / ddist_j;
				if (f_dist >= 0.8)
				{
					tis.voting_tag.at<uchar>(k, 0) = 0;
				}
			}
			else
			{
				tis.voting_tag.at<uchar>(k, 0) = 0;
			}			
		}
		//tis.vnnps = vnnps;
#if _DEBUG
		cout << tis.voting_tag << endl;
#endif
		(*vtis)[i] = tis;
	}
}

vector<testImgSift> SearchManager::TestImagePrepare(SearchParams params)
{
	vector<testImgSift> testList;
	FileManager fm;
	vector<string> testImgList = fm.FileReader(params.testImgFolder, "\\*.jpg");
	CV_Assert(testImgList.size() > 0);
	string gtfilename = params.testImgFolder + "\\gt.xml";
	cv::FileStorage fs(gtfilename, cv::FileStorage::READ);
	cv::FileNode node_image = fs["images"];
	cv::FileNodeIterator it_image_begin = node_image.begin();
	cv::FileNodeIterator it_image_end   = node_image.end();

	// --------REad ground truth from XMl file-------------//
	std::cout << "Read Ground Truth File..." << std::endl;
	for (; it_image_begin != it_image_end; it_image_begin++)
	{
		cv::FileNode node_id = *it_image_begin;
		cv::FileNodeIterator it_id_begin = node_id.begin();
		cv::FileNodeIterator it_id_end	 = node_id.end();

		testImgSift tis;
		tis.imgname = (*it_id_begin);
		it_id_begin++;
		tis.glat = (*it_id_begin);
		it_id_begin++;
		tis.glng = *it_id_begin;

		testList.push_back(tis);
	}

	std::cout << "Generate Feature Points..." << std::endl;
	ImageManager im;
	for (int i=0; i<testList.size(); i++)
	{
		testImgSift tis = testList[i];
		string imgName = params.testImgFolder + tis.imgname;
		cv::Mat imat = cv::imread(imgName);
		tis.descriptor = im.ImgSiftCollector(imat, false, 0.9);
		tis.voting_tag.create(tis.descriptor.rows, 1, CV_8UC1);
		tis.voting_tag.rowRange(0, tis.voting_tag.rows).setTo(1);
		testList[i] = tis;
		std::cout << "\r" << i << "%";
	}
	std::cout << "\n";
	return testList;
}

void SearchManager::TestImgTranslator(vector<testImgSift> *imgList, SearchParams params)
{
	CV_Assert(imgList->size() > 0);
	ImageManager im;
	string wdfile = params.wordFolder + "\\words.wd";
	string wdFlann = params.wordFolder + "\\words.flann";

	// Prepare word and its flann index
	cv::Mat words;
	FileManager fm;
	std::cout << "Read Word from disk..." << std::endl;
	fm.ReadMatFromDisk<float>(wdfile, &words);
	CV_Assert(words.data != 0);
	cv::flann::Index idx;
	idx.load(words, wdFlann);
	std::cout << "Translate image to word list..." << std::endl;
	for (int i=0; i<imgList->size(); i++)
	{
		cv::Mat indics, dists;
		idx.knnSearch((*imgList)[i].descriptor, indics, dists, 1);
		indics.convertTo(indics, CV_32FC1);
		(*imgList)[i].words = indics;
	}
}

// ------- The following two methods are working for Amir's framework
void SearchManager::AmirFlannSearch(vector<testImgSift> *vtis, SearchParams params)
{
	CV_Assert(vtis->size() > 0);
	FileManager fm;

	vector<string> flannList = fm.FileReader(params.amirFolder, "\\*.flann");
	CV_Assert(flannList.size() > 0);

	std::cout << "\nStart Doing Flann Searching..." << std::endl;

	int i, j, flannAmt;
	i = 0;
	flannAmt = flannList.size();
	// 1. Check output folder, if any existing result found, change i 
	vector<string> result_bfList = fm.FileReader(params.testImgFolder, "\\*.dist");
	if (result_bfList.size() == 0)
	{
		cout << "Without previous result, start from the first one..." << endl;
		i = 0;
		j = 0;
	}
	else
	{
		//std::cout << "Start from flann " << i << " image " << j << std::endl;
		string last_result = result_bfList[result_bfList.size() - 1];

		bf::path p(last_result);
		last_result = p.filename().string();

		vector<string> fmList;
		fm.file_name_splitter(last_result, &fmList, "_");



		string j_str = fmList[fmList.size() - 3];  // Image index number
		string i_str = fmList[fmList.size() - 2]; // Flann index number
		j = atoi(j_str.c_str()) + 1;
		if (j == vtis->size())
		{
			// All images are finished in previous flann index
			i = atoi(i_str.c_str()) + 1; //start from next flann index
			j = 0;
		}
		else
		{
			// previous flann search is stopped at img j
			i = atoi(i_str.c_str());
		}		
	}

	std::cout << " Start from flann_" << i << endl;

	for (; i < flannAmt; i++)
	{
		cv::flann::Index index;
		string flannFilename = flannList[i];
		string siftFilename = fm.FileExtensionChange(flannFilename, "sift");
		string gpsFilename = fm.FileExtensionChange(flannFilename, "gps");

		cv::Mat featureMat, _gpsMat;
		fm.ReadMatFromDisk<float>(siftFilename, &featureMat);
		fm.ReadMatFromDisk<double>(gpsFilename, &_gpsMat);

		index.load(featureMat, flannFilename);

		string zeros="0";			

		int n = (i == 0? 1:log10(i)+1);
		for (int z=0; z< 4-n; z++)
		{
			zeros = zeros + "0";
		}

		for (/*int j = 0*/; j < (*vtis).size(); j++)
		{
			testImgSift tis = (*vtis)[j];



			bf::path p(tis.imgname);

			string imgfilename = p.filename().string();
#if _DEBUG
			cout << p.filename().string() << endl;
#endif

			char buffer_indics[256], buffer_dist[256];

			string _zeros = "0";
			int _n = (j==0? 1: log10(j) + 1);
			for (int z = 0; z < 4 - _n; z++)
			{
				_zeros = _zeros + "0";
			}

			sprintf(buffer_indics, "%s\\%s%d_%s%d_%s.indics", params.testImgFolder.c_str(), _zeros.c_str(), j,  zeros.c_str(), i, imgfilename.c_str());
			sprintf(buffer_dist, "%s\\%s%d_%s%d_%s.dist", params.testImgFolder.c_str(), _zeros.c_str(), j, zeros.c_str(), i, imgfilename.c_str());

			cv::Mat indics, dists;
			double st = (double)cv::getTickCount();
			index.knnSearch(tis.descriptor, indics, dists, 5);
			st = ((double)cv::getTickCount() - st) / cv::getTickFrequency();

			string msg = ">>>>>>>>>>>>>>>>>>>>>>Flann Searching Time<<<<<<<<<<<<<<<<<<<<<<<\n";
			string logfilename = params.outputFolder + "\\time.log";
			std::cout << msg;
			//fm.WriteTimeLog(msg, logfilename);

			std::cout << "Eclipse " << st << "s" <<  std::endl;
			char logBuffer[256];
			sprintf(logBuffer, "%lf\n", st);
			fm.WriteTimeLog(static_cast<string>(logBuffer), logfilename);

			msg = ">>>>>>>>>>>>>>>>>>>>>>>>>>><<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<\n";
			std::cout << msg;
			indics.convertTo(indics, CV_32FC1);
#if _DEBUG
			cout << "save flann "<< i << "img " << j << "result";
#endif

			fm.SaveMat2Disk<float>(indics, static_cast<string>(buffer_indics));
			fm.SaveMat2Disk<float>(dists, static_cast<string>(buffer_dist));
		}
		j = 0;
		cout << "Flann_" << i << " is done...";
	}
}

void SearchManager::AmirFlannSearchMerge(vector<testImgSift> *vtis, SearchParams params)
{
	FileManager fm;
	vector<string> distList = fm.FileReader(params.testImgFolder, "\\*.dist");
	vector<string> indicsList = fm.FileReader(params.testImgFolder, "\\*.indics");
	CV_Assert(distList.size() > 0);
	CV_Assert(distList.size() == indicsList.size());

	int i=0;
	vector<string> gpsList = fm.FileReader(params.amirFolder, "\\*.gps");
	int flann_amt = gpsList.size();

	// 1. Prepare GPS mat data
	std::cout << "Prepare GPS vector..." << std::endl;
	vector<cv::Mat> vGps;

	for (int _i=0; _i<gpsList.size(); _i++)
	{
		cv::Mat _gps;
		fm.ReadMatFromDisk<double>(gpsList[_i], &_gps);
		vGps.push_back(_gps);
	}

	// 2. Prepare all distance and indics mat
	std::cout << "Prepare all distances and indics mat..." << std::endl;
	vector<cv::Mat> vDists, vIndics;
	for (int _i=0; _i<distList.size(); _i++)
	{
		cv::Mat dist, indics;
		fm.ReadMatFromDisk<float>(distList[_i], &dist);
		fm.ReadMatFromDisk<float>(indicsList[_i], &indics);
		vDists.push_back(dist);
		vIndics.push_back(indics);
	}

	// 3.Merge searching results
	std::cout << "Merging searching result..." << std::endl;
	for (int j=0; j < vtis->size(); j++) // For each image
	{
		testImgSift tis = (*vtis)[j];
		tis.vnnps.resize(tis.descriptor.rows);

		for (; i < (j+1)*flann_amt; i++)
		{
			cv::Mat dist, indics;
			dist = vDists[i];
			indics = vIndics[i];
			//fm.ReadMatFromDisk<float>(distList[i], &dist);
			//fm.ReadMatFromDisk<float>(indicsList[i], &indics);

			int flann_id = i % flann_amt;
			cv::Mat _gps = vGps[flann_id];
			//fm.ReadMatFromDisk<double>(gpsList[flann_id], &_gps);

			for (int k = 0; k < dist.rows; k++)
			{
				for (int d = 0; d < dist.cols; d++)
				{
					int id = flann_id * 5 + d;
					double _indics = indics.at<float>(k, d);
					double _dist = dist.at<float>(k, d);
					double _lat = _gps.at<double>(_indics, 0);
					double _lng = _gps.at<double>(_indics, 1);
					NNPoint nnp(id, _indics, _dist, _lat, _lng);
					tis.vnnps[k].insert(nnp);

#if 0
					printNNP_Set(tis.vnnps[k]);
#endif
				}
			}
		}

		(*vtis)[j] = tis;
		cout << "img_" << j << " is merged..." << endl;
	}
	Voting(vtis, params);
}

//-------Flann search for multiple groups-----------//
void SearchManager::FlannSearchMultipleGP(vector<vector<testImgSift>> v_groups, SearchParams params)
{
	CV_Assert(v_groups.size() > 0);
	FileManager fm;
	
	string sceneModelFolder = params.trGpFolder;	
	vector<string> scModelList = fm.FolderReader(sceneModelFolder + "\\features");
	CV_Assert(scModelList.size() > 0);

	// Prepare output folder to indics and distance files
	fm.DirectoryBuilder(params.outputFolder);
	bf::path _p(params.outputFolder);
	if (!bf::exists(_p))
	{
		params.outputFolder = params.outputFolder + "\\";
		fm.DirectoryBuilder(params.outputFolder);
	}

	int scene_id, flann_id, img_id;
	Preresult(params.outputFolder, &scene_id, &flann_id, &img_id);

	scene_id++;
	flann_id++;
	img_id++;

	for (int i=scene_id; i<v_groups.size() > 0; i++)
	{
		vector<testImgSift> v_tis = v_groups[i];

		if (v_tis.size() > 0)
		{
			vector<string> flannList = fm.FileReader(scModelList[i], "\\*.flann");
			FlannSearchMultipleGPPer(v_tis, flannList, scene_id, flann_id, img_id, params.outputFolder);
		}
	}
}

void SearchManager::FlannSearchMultipleGPPer(vector<testImgSift> vtis, vector<string> flannList, int scene_id, int flann_id, int img_id, string bkFolder)
{
	CV_Assert(vtis.size() > 0);
	CV_Assert(flannList.size() > 0);
	FileManager fm;
	
	string zero_scene = fm.zeros(4, scene_id);

	std::cout << "\n Start doing flann seaching..." << std::endl;
	for (int i=flann_id; i < flannList.size(); i++)
	{
		string flannFilename = flannList[i];
		string featureFilename = fm.FileExtensionChange(flannList[i], "sift");
		string gpsFilename = fm.FileExtensionChange(flannList[i], "gps");

		cv::Mat sift;
		fm.ReadMatFromDisk<float>(featureFilename, &sift);
		CV_Assert(sift.data != 0);

		cv::flann::Index index;
		index.load(sift, flannFilename);
		
		string zeros_flann = fm.zeros(4, i);

		for (int j=img_id; j<vtis.size(); j++)
		{
			testImgSift tis = vtis[j];
			cv::Mat indics, dists;
			double st = (double)cv::getTickCount();
			index.knnSearch(tis.descriptor, indics, dists, 5);
			st = ((double)cv::getTickCount() - st) / cv::getTickFrequency();
			indics.convertTo(indics, CV_32FC1);
			dists.convertTo(dists, CV_32FC1);

			string zero_img = fm.zeros(4, img_id);
			char indics_buffer[256];
			char dists_buffer[256];
			std::sprintf(indics_buffer, "%s\\%s%d_%s%d_%s%d.indics", bkFolder.c_str(), zero_scene.c_str(), scene_id, zeros_flann.c_str(), flann_id, zero_img.c_str(), img_id);
			std::sprintf(dists_buffer, "%s\\%s%d_%s%d_%s%d.dist", bkFolder.c_str(), zero_scene.c_str(), scene_id, zeros_flann.c_str(), flann_id, zero_img.c_str(), img_id);
		}		
	}
}

void SearchManager::Preresult(string bkFolder, int *scene_id, int *flann_id, int *img_id)
{
	FileManager fm;
	vector<string> preRsltList = fm.FileReader(bkFolder, "\\*.indics");
	if (preRsltList.size() > 0)
	{
		string last_indics = preRsltList[preRsltList.size() - 1];
		bf::path path(last_indics);
		string filename = path.filename().string();
		vector<string> fnList;
		fm.file_name_splitter(filename, &fnList, "_");

		string scene_str = fnList[fnList.size() - 3];
		string flann_str = fnList[fnList.size() - 2];
		string img_str = fnList[fnList.size() - 1];

		(*scene_id) = atoi(scene_str.c_str());
		(*flann_id) = atoi(flann_str.c_str());
		(*img_id) = atoi(img_str.c_str());
	}
	else
	{
		// Without previous result, start from the beginning 
		*scene_id = -1;
		*flann_id = -1;
		*img_id = -1;
	}
}