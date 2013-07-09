#include "WordUtilities.h"
#include "nemo_dll/WordModel.h"

using namespace nemo;

void WordUtilities::wordGeneration(string featureFolder, double clusterAmt)
{
	FileManager fm;
	WordModel wm;
	vector<string> featureList = fm.FileReader(featureFolder, "\\*.sift");
	
	CV_Assert(featureList.size() > 0);

	for (int i=0; i < featureList.size(); i++)
	{
		cv::Mat features;
		fm.ReadMatFromDisk<float>(featureList[i], &features);
		CV_Assert(features.data != 0);
		int _clustAmt = features.rows * clusterAmt;
		std::cout << _clustAmt << " words is generating...." << std::endl;

		cv::Mat words = wm.generateWordDictionary(features, _clustAmt);		
		std::cout << words.rows << " words is generated..." << std::endl;

		string wdFilename = fm.FileExtensionChange(featureList[i], "doc");
		fm.SaveMat2Disk<float>(words, wdFilename);
		std::cout << " Word " << i << std::endl;
	}
}

void WordUtilities::word_entropy_builder(string ref_imgFolder, string wdFilename, string wdflannfilename)
{
	bf::path path(wdFilename);
	string wdDir = path.parent_path().string();


	ImageManager im;
	FileManager fm;
	CubeMat cm = fm.CubeMatSampling(ref_imgFolder, 3);
	CV_Assert(cm.size() > 0);

	// Prepare word mat and flann index
	cv::Mat word;
	std::cout << "Read word file...\n";
	fm.ReadMatFromDisk<float>(wdFilename, &word);
	CV_Assert(word.data != 0);

	std::cout << "Read flann index...\n";
	cv::flann::Index wdidx;
	wdidx.load(word, wdflannfilename);

	vector<vector<long int > > word_invert(word.rows);
	cv::Mat mat_word_invert(0, word.rows,  CV_32SC1); // use mat transfer

#if 0
	cv::Scalar val(-1);
	cv::Mat _row(1, mat_word_invert.cols, mat_word_invert.type(), val);

	std::cout << _row(cv::Rect(0, 0, 10, 1)) << std::endl;
#endif
	std::cout << "Start invert analysis for " << cm.size() << " spots...\n";
	int store_i = 0;
	for(int i=0; i<cm.size(); i++)
	{
		string imgFilename = cm[i][0];
		bt::tuple<int, int> t_invert_file;

		int img_idx, view_idx;
		fm.imageIdxExtractor(imgFilename, &img_idx, &view_idx);
		t_invert_file.get<0>() = img_idx;

		cv::Mat global_indics(0, 1, CV_32SC1);
		for(int j=0;j<cm[i].size();j++)
		{
			imgFilename = cm[i][j];
			cv::Mat imat = cv::imread(imgFilename);
			cv::Mat descriptor = im.ImgSiftCollector(imat, false, 0.9);
			CV_Assert(descriptor.data != 0);

			cv::Mat indics, distance;
			wdidx.knnSearch(descriptor, indics, distance, 1);
			global_indics.push_back(indics);
		}

		// summarize GPS location for each word
		invert_file(global_indics, img_idx, &word_invert);
		int globalSize = vec_invert_filesize(&word_invert);
		if (globalSize > NEMO_GB * 1.5)
		{			
			char buffer[256];
			string _zeros = fm.zeros(4, store_i);
			sprintf(buffer, "%s\\%s%d.wdFreq", wdDir.c_str(), _zeros.c_str(), store_i);
			save_word_invert(word_invert, static_cast<string>(buffer));
			word_invert.clear();
			store_i++;
		}

#if 0
		invert_file_mat(global_indics, img_idx, &mat_word_invert);
		double descriptorsSize = mat_word_invert.rows * mat_word_invert.cols * sizeof(float);
		if (descriptorsSize > NEMO_GB * 1.5)
		{
			std::cout << "\nLarge data file, back it up, check your word folder...\n";
			char buffer[256];
			string _zeros = fm.zeros(4, store_i);
			sprintf(buffer, "%s\\%s%d.wdFreq", wdDir.c_str(), _zeros.c_str(), store_i);
			fm.SaveMat2Disk<float>(mat_word_invert, static_cast<string>(buffer));

			store_i++;
			mat_word_invert.release();
		}
#endif

		printf("\r%d%% is processed", i);
	}	

	char _buffer[256];
	string _zeros = fm.zeros(4, store_i);
	sprintf(_buffer, "%s\\%s%d.wdFreq", wdDir.c_str(), _zeros.c_str(), store_i);
	save_word_invert(word_invert, static_cast<string>(_buffer));
}

void WordUtilities::invert_file(cv::Mat indics, int imgId, vector<vector<long int > > *word_invert)
{
	vector<bt::tuple<long int, long int> > v_wf;
	
	v_wf = word_freq(indics);
	for (int i=0; i<v_wf.size(); i++)
	{
		bt::tuple<int, int> t_word_freq = v_wf[i];
		int wd_id = t_word_freq.get<0>();
		int wd_freq = t_word_freq.get<1>();

		//create a tuple
		bt::tuple<int, int> t_img_freq(imgId, wd_freq);
		//(*word_invert)[wd_id].push_back(t_img_freq);
		(*word_invert)[wd_id].push_back(imgId);
		(*word_invert)[wd_id].push_back(wd_freq);
	}
}

void WordUtilities::invert_file_mat(cv::Mat indics, int imgId, cv::Mat *m_invert_file)
{
	CV_Assert(indics.data != 0);
	
	cv::Scalar val(-1);
	cv::Mat _row(2, m_invert_file->cols, m_invert_file->type(), val);

	vector<bt::tuple<long int, long int> > v_word_freq = word_freq(indics);
	for (int i=0; i < v_word_freq.size(); i++)
	{
		int wd_id = v_word_freq[i].get<0>();
		int wd_freq = v_word_freq[i].get<1>();
		_row.at<int>(0, wd_id) = imgId;		
		_row.at<int>(1, wd_id) = wd_freq;
	}
#if 0
	std::cout << _row <<std::endl;
#endif
	m_invert_file->push_back(_row);
}

int WordUtilities::vec_invert_filesize(vector<vector<long int > > *word_invert)
{
	int globalSize = 0;
	for (int i=0; i<word_invert->size(); i++)
	{
		int _length = (*word_invert)[i].size();
		globalSize += _length;
	}
	globalSize = globalSize *(2 * sizeof(int));
	return globalSize;
}

vector<bt::tuple<long int, long int> > WordUtilities::word_freq(cv::Mat _indics)
{
	CV_Assert(_indics.data != 0);

	Word_FREQ wf;

	for(int i=0; i<_indics.rows; i++)
	{
		int wdIdx = _indics.at<int>(i, 0);

		voc_word vw(wdIdx, 1);
		Word_FREQ::iterator wf_it = wf.find(vw);

		if(wf_it == wf.end())
		{
			wf.insert(vw);
		}
		else
		{
			voc_word temp_vw = *wf_it;
			temp_vw.freq++;
			wf.replace(wf_it, temp_vw);
		}       
	}

	Word_FREQ::iterator _it = wf.begin();
	vector<bt::tuple<long int, long int> > v_wf;
	for(; _it != wf.end(); ++_it)
	{
		voc_word _temp = *_it;
		v_wf.push_back(bt::make_tuple(_temp.wid, _temp.freq));
#if 0
		if(_temp.freq > 1)
			std::cout << _temp.wid << "\t" << _temp.freq << std::endl;
#endif
	}   

	return v_wf;
}

// Save word_invert vector into a matrix. 
void WordUtilities::save_word_invert(vector<vector<long int > > word_invert, string filename)
{
	cv::Mat _cols(0, 1, CV_32SC1);
	cv::Mat _cols_idx(1, word_invert.size(), CV_32SC1);
	int pre_length=0;
	for (int i=0; i< word_invert.size(); i++)
	{
		// every word
		vector<long int> _data = word_invert[i];
		
		if (_data.size() > 0)
		{
			cv::Mat _cols_data(_data.size(), 1, CV_32SC1, _data.data());
			_cols.push_back(_cols_data);
			//_cols_idx.at<int>(0, i) = _data.size();
			pre_length += _data.size();
		}
		//else
		{
			_cols_idx.at<int>(0, i) = pre_length; // store index
		}				
	}
	
	FileManager fm;
	fm.SaveMat2Disk<int>(_cols, filename);
	string idxFilename = fm.FileExtensionChange(filename, "idx");
	fm.SaveMat2Disk<int>(_cols_idx, idxFilename);
}

vector<vector<long int > > WordUtilities::read_word_invert(string filename)
{
	bf::path path(filename);
	if (!bf::exists(path))
	{
		std::cout << "Cannot find invert word file..." << std::endl;
		exit(-1);
	}

	FileManager fm;
	string idx_filename = fm.FileExtensionChange(filename, "idx");
	bf::path idx_path(idx_filename);
	if (!bf::exists(idx_path))
	{
		std::cout << "Cannot find invert file index file..." << std::endl;
		exit(-1);
	}

	cv::Mat mat_invert_word, mat_invert_idx;
	fm.ReadMatFromDisk<int>(filename, &mat_invert_word);
	fm.ReadMatFromDisk<int>(idx_filename, &mat_invert_idx);

	vector<vector<long int > > v_invert_word(mat_invert_idx.cols);

	int start_row=0;
	for (int i=0; i<mat_invert_idx.cols; i++)
	{
		int row_length = mat_invert_idx.at<int>(0, i);
		cv::Mat wordData = mat_invert_word(cv::Rect(0, start_row, 1, row_length - start_row));
#if 0
		std::cout << wordData << std::endl;
#endif		
		
		v_invert_word[i].assign((long int*)wordData.data, (long int*)wordData.data+ wordData.rows);  //(row_length-start_row)
		
		if (row_length != 0)
		{
			start_row = row_length;
		}		
	}

	return v_invert_word;	
}

void WordUtilities::invert_file_analysis(string invertFilename, string gpsFilename)
{	
	FileManager fm;
	
	vector<vector<long int > > invert_file = read_word_invert(invertFilename);
	CV_Assert(invert_file.size() > 0);
		
	cv::Mat gpsMat, centerMat(0, 2, CV_64FC1);
	fm.GpsDataReader(gpsFilename, &gpsMat);
	int imgAmt = gpsMat.rows;

	vector<float> v_word_radius(invert_file.size());
	cv::Mat invert_file_entro(invert_file.size(), 4, CV_64FC1);

	for (int i=0; i<invert_file.size(); i++)
	{
		//vector<cv::Point2f> points;
		cv::Mat points(0, 2, CV_64FC1);
		for (int j=0; j < invert_file[i].size(); j+=2)
		{
			int imgId = invert_file[i][j];
			double alti = gpsMat.at<double>(imgId, 1);
			double lngi = gpsMat.at<double>(imgId, 2);
			cv::Mat mpt(1, 2, CV_64FC1);
			mpt.at<double>(0, 0) = alti;
			mpt.at<double>(0, 1) = lngi;
			points.push_back(mpt);
		}

		if (invert_file[i].size() > 0)
		{
			invert_file_entro.at<double>(i, 3) = log(imgAmt / invert_file[i].size());
		}
		

		if (points.rows > 0)
		{
			cv::Mat centers;
			float radius;
			//cv::minEnclosingCircle(points, centers, radius);

			cv::Mat means_alti, means_lngi, stdv_alti, stdv_lngi;
#if _DEBUG
			std::cout << points << std::endl;
#endif
			//cv::Mat m_alti = mPts(cv::Rect(0,0, ))
			cv::Mat p_alti(points.col(0)), p_lngi(points.col(1));
			cv::meanStdDev(p_alti, means_alti, stdv_alti);
			cv::meanStdDev(p_lngi, means_lngi, stdv_lngi);
			cv::pow(stdv_alti, 2.0, stdv_alti);
			cv::pow(stdv_lngi, 2.0, stdv_lngi);
			cv::Mat dst;
			cv::add(stdv_alti, stdv_lngi, dst);
			cv::sqrt(dst, dst);
			
#if _DEBUG
			std::cout << dst << std::endl;
			std::cout << means_alti << std::endl;
			std::cout << means_lngi << std::endl;
			if (dst.at<double>(0, 0) < 0)
			{
				std::cout << "Check here" << std::endl;
			}
#endif
			//check mean and stdv should be 1 x 1 matrix
			CV_Assert(dst.cols * dst.rows == 1);
			CV_Assert(means_alti.cols * means_alti.rows == 1);
			CV_Assert(means_lngi.cols * means_lngi.rows == 1);
			CV_Assert(stdv_alti.cols * stdv_alti.rows == 1);
			CV_Assert(stdv_lngi.cols * stdv_lngi.rows == 1);

			invert_file_entro.at<double>(i, 0) = dst.at<double>(0,0);
			//invert_file_entro.at<double>(i, 1) = means_alti.at<double>(0, 0);
			//invert_file_entro.at<double>(i, 2) = means_lngi.at<double>(0, 0);
			invert_file_entro.at<double>(i, 1) = stdv_alti.at<double>(0, 0);
			invert_file_entro.at<double>(i, 2) = stdv_lngi.at<double>(0, 0);
		}
		else
		{
			//v_word_radius[i] = 0;
			//cv::Point2f center(0, 0);
			//centerMat.push_back(center);
			invert_file_entro.at<double>(i, 0) = 0;
			invert_file_entro.at<double>(i, 1) = 0;
			invert_file_entro.at<double>(i, 2) = 0;
			invert_file_entro.at<double>(i, 3) = 0;
		}
	}
	
	std::cout << invert_file_entro.row(3) << std::endl;
	string radius_filename = fm.FileExtensionChange(invertFilename, "radius");
	fm.SaveMat2Disk<double>(invert_file_entro, radius_filename);
	std::cout << "Radius save finished...\n" ;
}

void WordUtilities::findMinEnclosedCircle(vector<cv::Point2f> v_points, cv::Mat *center, float *radius)
{
	CV_Assert(v_points.size() > 0);
	cv::Mat m_Points(v_points);
}

void WordUtilities::search_by_word(string wdfilename, string testImgFolder, string invert_filename, string gpsfilename, int type, string ref_imgFolder)
{
	FileManager fm;
	ImageManager im;
	std::cout << "Read test images from" << testImgFolder << "... " << std::endl;
	vector<string> testimgList = fm.FileReader(testImgFolder, "\\*.jpg");
	
	if (testimgList.size() == 0)
	{
		generateModelTestImages(ref_imgFolder, gpsfilename, 100, testImgFolder);
		testimgList = fm.FileReader(testImgFolder, "\\*.jpg");
	}

	struct gps_loc
	{
		double lati;
		double lngi;
	};
	vector<gps_loc> gpsList(testimgList.size());

	CV_Assert(testimgList.size()>0);	

	cv::Mat gpsMat;
	fm.GpsDataReader(gpsfilename, &gpsMat);
	CV_Assert(gpsMat.data !=0 );
	for (int i=0; i<testimgList.size(); i++)
	{
		int imgIdx, viewIdx;
		fm.imageIdxExtractor(testimgList[i], &imgIdx, &viewIdx);

#if _DEBUG
		std::cout << imgIdx << std::endl;
#endif

		gpsList[i].lati = gpsMat.at<double>(imgIdx, 1);
		gpsList[i].lngi = gpsMat.at<double>(imgIdx, 2);
	}
	

#if _DEBUG
	vector<int> _v;
	_v.push_back(675);
	save_result_images(_v, testimgList[0]);
#endif	

	cv::Mat words;
	std::cout << "Read word files ..." << std::endl;
	fm.ReadMatFromDisk<float>(wdfilename, &words);
	string wdIdxFilename = fm.FileExtensionChange(wdfilename, "flann");

	std::cout << " Read word index ..." << std::endl;
	cv::flann::Index wdIdx;
	wdIdx.load(words, wdIdxFilename);

	// Prepare inver-file 
	std::cout << "Read invert file ..." << std::endl;
	vector<vector<long int> > v_invert_file = read_word_invert(invert_filename);	
	string radius_filename = fm.FileExtensionChange(invert_filename, "radius");
	cv::Mat radius_mat;
	fm.ReadMatFromDisk<double>(radius_filename, &radius_mat);

	std::cout << "Start doing vocabulary tree searching ... " << std::endl;
	int corr_count = 0;
	double total_time = 0;
	for (int i=0; i<testimgList.size(); i++)
	{
		cv::Mat imat = cv::imread(testimgList[i]);
		if (imat.data != 0)
		{
			cv::Mat descriptor = im.ImgSiftCollector(imat, false, 0.9);
			CV_Assert(descriptor.data != 0);
			cv::Mat indics, dists;
			wdIdx.knnSearch(descriptor, indics, dists, 1); // map descriptor to words

			// GPS location!-- image id of winer 
			double start_time = (double)cv::getTickCount();
			vector<int> _v = vote_by_word(indics, v_invert_file, radius_mat, type);
			start_time = ((double)cv::getTickCount() - start_time)/cv::getTickFrequency();
			total_time = total_time + start_time;
			std::cout << "Eclipse time: " << start_time << std::endl;
			std::cout << "Save result images ..." << std::endl;
			
			
			int target_idx = _v[0];
			double lat = gpsMat.at<double>(target_idx, 1);
			double lng = gpsMat.at<double>(target_idx, 2);

			double gps_dist = sqrt(pow((gpsList[i].lati - lat), 2.0) + pow((gpsList[i].lngi - lng), 2));
			std::cout << i << ":$" << testimgList[i] << "\n" << target_idx << "\t" << gps_dist << "\n";
			if (gps_dist < 0.0005)
			{
				corr_count++;
			}
			//save_result_images(_v, testimgList[i]);
		}
	}

	float per_corr = (float)corr_count / (float)testimgList.size();
	std::cout << "Correct result " << per_corr << " \n Searching Finish ...\n Eclipse " << total_time << std::endl;
}

vector<int> WordUtilities::vote_by_word(cv::Mat indics, vector<vector<long int > > v_invert_file, cv::Mat radius_mat, int type)
{
	CV_Assert(indics.data != 0);
	FileManager fm;
	
	Img_FREQ wf;
	

	for (int i=0; i<indics.rows; i++)
	{
		int wdIdx = indics.at<int>(i, 0);
		//std::cout << "word index:" << wdIdx << std::endl;
		double radius = radius_mat.at<double>(wdIdx, type); // radius: 0 - radius, 1- alti, 2 - lngi, 3 - freq
		//std::cout << "word radius is " << radius << std::endl;

		vector<long int> imgList_wd = v_invert_file[wdIdx];
	
	for (int j=0; j<imgList_wd.size(); j+=2)
		{
			long int img_Id = imgList_wd[j];
			long int img_Freq = imgList_wd[j+1];
			
			img_word vw(img_Id, img_Freq);
			Img_FREQ::iterator wf_it = wf.find(vw);
			if (wf_it == wf.end())
			{
				wf.insert(vw);
			}
			else
			{
				img_word temp_vw = *wf_it;
				if (radius != 0)
				{
					temp_vw.freq += (vw.freq* 1.0/radius);
					wf.replace(wf_it, temp_vw);
				}	
				else
				{
					temp_vw.freq += 0;
					wf.replace(wf_it, temp_vw);
				}
			}
		}
	}
	
	vector<int> _v;
	Img_FREQ::nth_index<2>::type& wf_idx = wf.get<2>();
	Img_FREQ::nth_index<2>::type::iterator wf_it = wf_idx.end();
	int k=0;
#if 1
	for (wf_it--; k<5; wf_it--,k++)
	{
		//std::cout << (*wf_it).imgId << std::endl;
		//std::cout << (*wf_it).freq << std::endl;
		_v.push_back((*wf_it).imgId);
	}
#endif	
	//std::cout << (*wf_it).imgId << std::endl;

	return _v;
}

void WordUtilities::save_result_images(vector<int> _v, string imgName)
{
	//CV_Assert(_v.size() > 0);
	size_t position = imgName.find(".");
	string extractName = (string::npos == position)? imgName:imgName.substr(0, position);
	FileManager fm;
	string str = extractName;
	string dst_folder = fm.DirectoryBuilder( str + "\\");
	int imgAmt = 250;
	string imgFolder = "C:\\TDDownload\\GoogleStreetview\\1210724640000\\DVD V1.7 Pittsburgh\\1210724640000\\images\\";
	for (int i=0; i<_v.size(); i++)
	{
		int imgId = _v[i];		
		string zs = "";
		for (int k=0; k<4-log10(imgId); k++)
		{
			zs += "0";
		}

		int folderIdx = imgId / imgAmt;
		string f_zs = "";
		for (int k=0; k < 1-log10(folderIdx); k++)
		{
			f_zs += "0";
		}		
		
		for (int j=0; j<4; j++)
		{
			char buffer_src[256];
			sprintf(buffer_src, "%s%s%d\\reprojected_%s%d_%d.jpg", imgFolder.c_str(), f_zs.c_str(), folderIdx, zs.c_str(), imgId, j);
			string src = buffer_src;
			//std::cout << src << std::endl;
			char buffer_dst[256];
			sprintf(buffer_dst, "%sreprojected_%s%d_%d_%d.jpg", dst_folder.c_str(), zs.c_str(), imgId, j, i);
			string dst = buffer_dst;

			fm.copyFile(src, dst);
		}				
	}
	
}

void WordUtilities::generateModelTestImages(string ref_imgfolder, string gpsfilename, int imgAmt, string testImgFolder)
{
	FileManager fm;
	CubeMat cm = fm.CubeMatSampling(ref_imgfolder, 1);
	CV_Assert(cm.size() > 0);

	vector<TstImg> testImgList;

	cv::Mat gpsMat;
	fm.GpsDataReader(gpsfilename, &gpsMat);
	CV_Assert(gpsMat.data != 0);
	
	std::cout << "Generate " << imgAmt << " test images.. ";
	for (int i=0; i< imgAmt; i++)
	{
		int cube_number = rand()%(int)(cm.size());
		int view_number = rand()%(int)(4);

		string imgName = cm[cube_number][view_number];
		bf::path p(imgName);
		string dstName = testImgFolder + "\\" + p.filename().string();
		fm.copyFile(imgName, dstName);
		std::cout << "\r"<< i ;
		TstImg timg;
		timg.filename = p.filename().string();
		timg.lat = gpsMat.at<double>(cube_number, 1);
		timg.lng = gpsMat.at<double>(cube_number, 2);

		testImgList.push_back(timg);
	}

	fm.GenerateGTXmlFile(testImgList, testImgFolder + "\\gt.xml");
}