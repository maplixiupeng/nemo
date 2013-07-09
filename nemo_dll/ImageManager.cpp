#include "ImageManager.h"
#include "LogManager.h"

using namespace nemo;

cv::Mat ImageManager::ImgSiftCollector(cv::Mat imat, bool isDisp /* = false */, double sig)
{
	CV_Assert(imat.data != 0);
	
	cv::Mat gray_imat; 
	if (imat.channels() == 1)
	{
		imat.copyTo(gray_imat);
	}
	else
	{
		cv::cvtColor(imat, gray_imat, CV_BGR2GRAY);
	}

	cv::SiftFeatureDetector detector(400, 3, 0.04,10, sig);
	//cv::SiftFeatureDetector detector;
	vector<cv::KeyPoint> kps;
	//cv::blur(gray_imat, gray_imat, cv::Size(10, 10));
	detector.detect(gray_imat, kps);
	
	cv::SiftDescriptorExtractor extractor;
	cv::Mat descripor;
	extractor.compute(gray_imat, kps, descripor);

	if (isDisp)
	{
		DrawSurfFeature(imat, kps);
	}

	return descripor;
}

//void ImageManager::ImgSiftCollectorFrFolder(vector<string> imgFolder, cv::Mat gps_mat)
//{
//	CV_Assert(imgFolder.size() > 0);
//	CV_Assert(gps_mat.data != 0);
//
//	FileManager fm;
//	
//	for (int i=0; i < imgFolder.size(); i++)
//	{
//		vector<string> folder = fm.FileReader(imgFolder[i], "\\*.jpg");
//		cv::Mat descriptors_gps(0,2,CV_64FC1);
//		cv::Mat descriptors;
//
//		for (int j =0; j < folder.size(); j++)
//		{
//			string imgFilename = folder[j];
//			cv::Mat imat = cv::imread(imgFilename);
//			imat = ImageClipper(imat);
//
//			SiftGPS sg;
//			sg.featureVector = ImgSiftCollector(imat, false, 0.9);
//
//			int imgIdx, viewIdx;
//			fm.imageIdxExtractor(imgFilename, &imgIdx, &viewIdx);
//			double lat = gps_mat.at<double>(imgIdx, 1);
//			double lng = gps_mat.at<double>(imgIdx, 2);			
//
//			descriptors.push_back(sg.featureVector);
//
//			cv::Mat _gps(sg.featureVector.rows, 2, CV_64FC1);
//			_gps.colRange(0, 1).setTo(lat);
//			_gps.colRange(1, 2).setTo(lng);
//
//			descriptors_gps.push_back(_gps);
//		}
//
//	}
//}

cv::Mat ImageManager::ImgSurfCollector(cv::Mat imat, bool isDisp /* = false */)
{
	CV_Assert(imat.data != 0);

	cv::Mat gray_imat;
	if (imat.channels() == 1)
	{
		imat.copyTo(gray_imat);
	}
	else
	{
		cv::cvtColor(imat, gray_imat, CV_BGR2GRAY);
	}

	cv::SurfFeatureDetector detector;
	vector<cv::KeyPoint> kps;
	detector.detect(gray_imat, kps);

	cv::SurfDescriptorExtractor extractor;
	cv::Mat descriptor;
	extractor.compute(gray_imat, kps, descriptor);

	if (isDisp)
	{
		DrawSurfFeature(imat, kps);
	}
	return descriptor;
}

cv::Mat ImageManager::ImageClipper(cv::Mat imat)
{
	CV_Assert(imat.data != 0);
	cv::Range range(0, 2*imat.rows/3);
	cv::Mat clip_imat(imat, range);
	//cv::imshow("Clip imat", clip_imat);
	//cv::waitKey(0);
	return clip_imat;
}

void ImageManager::DrawSurfFeature(cv::Mat imat, vector<cv::KeyPoint> keypoints)
{
	cv::RNG rng(12345);
	for (int i=0;i<keypoints.size();i++)
	{
		cv::KeyPoint kp = keypoints[i];
		cv::Scalar color = cv::Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		cv::circle(imat, kp.pt, 4, color);
	}
	cv::resize(imat, imat, cv::Size(imat.cols/2, imat.rows/2));
	cv::imshow("SURF", imat);
	cv::waitKey(0);
	return;
}

void ImageManager::FlannBuilder(CubeMat cm, string *flann_filename, cv::Mat mgps, string flann_folder)
{
	FileManager fm;
	
	cv::Mat flann_data(0, 128, CV_32F);
	cv::Mat gps_data(0, 2, CV_64FC1);

	int start_row = 0;

	for (int i=cm.size()/2; i < cm.size(); i++)
	{
		vector<string> simg_cube = cm[i];
		SiftCube _sc;
		vector<string> _v_imgName;
		fm.file_name_splitter(simg_cube[0], &_v_imgName, "_");
		int id = atoi(_v_imgName[1].c_str());
		_sc.id = id;
		double lat = mgps.at<double>(id, 1);
		double lng = mgps.at<double>(id, 2);

		for (int j=0; j<simg_cube.size(); j++)
		{			
			cv::Mat imat = cv::imread(simg_cube[j]);
			imat = ImageClipper(imat);
			cv::Mat descriptor = ImgSiftCollector(imat, false);

			//some image cannot detect feature points
			if (descriptor.rows > 0)
			{
				int rows = descriptor.rows;

				flann_data.resize( (size_t)start_row + (size_t)rows);
				descriptor.copyTo(flann_data(cv::Rect(0, start_row, descriptor.cols, descriptor.rows)));			

				gps_data.resize((size_t)start_row + (size_t)rows);
				cv::Mat _gps_data_alt(rows,1,CV_64FC1), _gps_data_lng(rows, 1, CV_64FC1);
				_gps_data_alt.colRange(0,1).setTo(lat);
				_gps_data_lng.colRange(0,1).setTo(lng);
				_gps_data_alt.copyTo(gps_data(cv::Rect(0, start_row, 1, rows)));
				_gps_data_lng.copyTo(gps_data(cv::Rect(1, start_row, 1, rows)));
				//gps_data.setTo(lat, mask(Rect(0, start_row, 1, 1)));
				//gps_data.setTo(lng, mask(Rect(1, start_row, 1, 1)));

				//cout << flann_data.row(start_row) << endl;
				//cout << gps_data.row(start_row) << endl;

				start_row = flann_data.rows;		
			}				
		}
		printf("\r%d Image cube is done", i);
	}
	
	fm.DirectoryBuilder(flann_folder+"\\");
	
	char buff_gps_filename[256];
	char buff_surf_filename[256];
	sprintf(buff_gps_filename, "%s\\%s.gps", flann_folder.c_str(), flann_filename->c_str());
	sprintf(buff_surf_filename, "%s\\%s.sift", flann_folder.c_str(), flann_filename->c_str());
	fm.SaveMat2Disk<float>(flann_data, static_cast<string>(buff_surf_filename));
	fm.SaveMat2Disk<double>(gps_data, static_cast<string>(buff_gps_filename));

	FlannBuilder(flann_data, *flann_filename, flann_folder);
}

void ImageManager::FlannBuilder(cv::Mat train_data, string flann_filename, string flann_folder)
{
	//cv::flann::KMeansIndexParams params;
	cout << "Building Flann" << endl;
	//cv::flann::AutotunedIndexParams params(0.8, 0.01,0, 0.05);
	cv::flann::KMeansIndexParams params(64);
	cv::flann::Index index(train_data, params);

	FileManager fm;
	string flann_dir = flann_folder + "\\";
	fm.DirectoryBuilder(flann_dir);

	//flann_filename = flann_dir + flann_filename;
	index.save(flann_filename);
	
	//index.save("flann_4185");
}

//Flann Retrieve
string ImageManager::FlannSearch(string img_filename, string flann_folder, cv::Mat imat)
{
	FileManager fm;
	LogManager lm;
	vector<string> flann_list_sift = fm.FileReader(flann_folder, "\\*.sift");
	cv::Mat descriptor = ImgSiftCollector(imat, true);
	vector<NNP_Set> _v_nnps(descriptor.rows);

	int start_id = 0;
	for (int i=0; i < flann_list_sift.size(); i++)
	{
		string flann_data_surf = flann_list_sift[i];
		printf("\nSearch in %s", flann_data_surf.c_str());

		vector<string> flann_namelist;
		fm.file_name_splitter(flann_data_surf, &flann_namelist, ".");
		string flann_prefix = flann_namelist[0];
		string flann_idx = flann_prefix + ".flann";
		string flann_data_gps = flann_prefix + ".gps";

		/*FileStorage fs(flann_data, FileStorage::READ);
		Mat m_flann_data, m_gps_data;
		fs["flann"] >> m_flann_data;
		fs["gps"] >> m_gps_data;*/
		cv::Mat m_surf_data, m_gps_data;
		double tick = (double)cv::getTickCount();

		fm.ReadMatFromDisk<float>(flann_data_surf, &m_surf_data);
		tick = ((double)cv::getTickCount() - tick) / cv::getTickFrequency();

		fm.ReadMatFromDisk<double>(flann_data_gps, &m_gps_data);

		
		cv::flann::Index index;
		index.load(m_surf_data, flann_idx);
		
		cv::Mat cur_indics, cur_dist;
		
		tick = ((double)cv::getTickCount() - tick) / cv::getTickFrequency();
		index.knnSearch(descriptor, cur_indics, cur_dist, NEMO_KNN_NEIG);
		tick = ((double)cv::getTickCount() - tick) / cv::getTickFrequency();
		cout << "\n\nSearch in Flann: " << i << "Done. " << tick << "s" << endl;
		char buffer[128];
		sprintf(buffer, "Search in Flann: %d Done. Elipse time is: %f", i, tick);
		//lm.TimeLogRecorder("FLANN_SEARCH_LOG.log", static_cast<string>(buffer), "FlannSearch", "Flann", tick);
		MergeNNP(&_v_nnps, cur_indics, cur_dist, &start_id, &m_gps_data);		
	}

	int mat_width = flann_list_sift.size() * 5 * 4; // every flann reture 5 nearest neighbor, every neighbor has 4 attrib: index, dist, lat, lng, 
	int mat_height = _v_nnps.size();
	vector<string> _filename;
	fm.file_name_splitter(img_filename, &_filename, "\\");

	string rst_filename = _filename[_filename.size()-1] + ".nemo";

	SaveFlannResult(rst_filename, _v_nnps, mat_width, mat_height);

	return rst_filename;
}

/*!
	\fn ImageManager::FlannSearch
Parameters:
	@param: string imgfolder
	@param: string flann_folder
	@param: int tree_idx
return:
	void
Do Flann search for several images. One flann tree is loaded and search all images
*/
string ImageManager::FlannSearch(string imgfolder, string flann_folder, int tree_idx, cv::Mat *gps_mat, bool isSift)
{
	FileManager fm;
	vector<string> img_List, flann_list;
	img_List = fm.FileReader(imgfolder, "\\*.jpg");
	flann_list = fm.FileReader(flann_folder, "\\*.flann");
	//CV_Assert(img_List.size() > 0);
	CV_Assert(flann_list.size() > 0);
	CV_Assert(flann_list.size() > tree_idx);

	vector<string> filenamelist;
	fm.file_name_splitter(flann_list[tree_idx], &filenamelist, ".");
	string flann_data_filename = filenamelist[0] + ".sift";
	string flann_gps_filename = filenamelist[0] + ".gps";

	cv::Mat m_sift_data, m_gps_data;
	fm.ReadMatFromDisk<float>(flann_data_filename, &m_sift_data);
	fm.ReadMatFromDisk<double>(flann_gps_filename, &m_gps_data);

	cv::flann::Index index;
	index.load(m_sift_data, flann_list[tree_idx]);
	
	string rst_folder = fm.DirectoryBuilder(imgfolder + "\\result\\");

	for (int i=0; i<img_List.size(); i++)
	{
		vector<string> _vfilename;
		fm.file_name_splitter(img_List[i], &_vfilename, "\\");

		string rst_img_folder = imgfolder + "\\result\\" + _vfilename[_vfilename.size()-1] + "\\";
		string rst_folder=fm.DirectoryBuilder(rst_img_folder);		

		cv::Mat cur_indics, cur_dist;
		cv::Mat imat = cv::imread(img_List[i]);
		imat = ImageClipper(imat);
		CV_Assert(imat.data != 0);

		cv::Mat descriptor;
		if (isSift)
		{
			descriptor = ImgSiftCollector(imat); 
		}
		else
		{
			descriptor = ImgSurfCollector(imat);
		}
		 
		index.knnSearch(descriptor, cur_indics, cur_dist, NEMO_KNN_NEIG);
		cout << "Image " << i << "is done!" << endl;

		string rstFilename = _vfilename[_vfilename.size()-1] + ".nemo";
		
		char buffer[256];
		sprintf(buffer, "%s%s_%d",rst_folder.c_str(), rstFilename.c_str(), tree_idx);
		rstFilename = static_cast<string>(buffer);

		fm.SaveMat2Disk<float>(cur_dist, rstFilename + ".dist");
		fm.SaveMat2Disk<int>(cur_indics, rstFilename + ".index");
	}

	(*gps_mat) = m_gps_data;
	return rst_folder;
}

void ImageManager::MergeNNP(vector<NNP_Set> *orig, cv::Mat indices, cv::Mat dists, int *start_id, cv::Mat *gps_data)
{
	CV_Assert(indices.rows == dists.rows);
	CV_Assert(indices.cols == dists.cols);

	if(!gps_data->data)
	{
		return;
	}

	for (int i=0; i<indices.rows; i++)
	{
		for (int j=0; j<indices.cols; j++)
		{
			int index = indices.at<int>(i, j);
			double dist = (double)dists.at<float>(i,j);
			double alt = gps_data->at<double>(index, 0);
			double lng = gps_data->at<double>(index, 1);

			NNPoint nnp((*start_id), index, (double)dist, alt, lng);
			(*orig)[i].insert(nnp);
			(*start_id)++;
		}
	}
}

void ImageManager::SaveFlannResult(string filename, vector<NNP_Set> _v_nnps, int mat_width, int mat_height)
{
	cv::Mat output;
	if (mat_width == 0 || mat_height == 0)
	{
		//output.create(_v_nnps.size(), 20*3, CV_32F);
		output.create(_v_nnps.size(), 20*3, CV_64FC1);
	} 
	else
	{
		//output.create(mat_height, mat_width, CV_32F);
		output.create(mat_height, mat_width, CV_64FC1);
	}
	//3 for dist, alt, lng

	for (int i=0; i<_v_nnps.size(); i++)
	{
		NNP_Set nnps = _v_nnps[i];
		const NNP_Set::nth_index<1>::type &distance = nnps.get<1>();	
		bm::nth_index<NNP_Set, 1>::type::iterator it = distance.begin();

		int k=0;
		for (; it != get<1>(nnps).end(); it++)
		{
			int index = it->indics;
			double dist = it->dist;
			double alt = it->alt;
			double lng = it->lng;

			//cout << "\rdist:\t" << dist << "alt:\t" << alt << "lng:\t" << lng << "k:\t" << k << endl;
			output.at<double>(i, k) = (double)index;
			output.at<double>(i, ++k) = dist;
			output.at<double>(i,++k) = alt;
			output.at<double>(i,++k) = lng;
			k++;
		}
	}

	FileManager ufp;
	ufp.SaveMat2Disk<double>(output, filename);
	MatlabMatGenerator(output, filename+".mat");
}

int ImageManager::MatlabMatGenerator(cv::Mat data_m, string filename)
{
	MATFile *pmat;
	mxArray *pa2;

	if (filename == "")
	{
		filename = "mattest.mat";
	}

	data_m = data_m.t();

	int rows = data_m.rows;
	int cols = data_m.cols;
	pa2 = mxCreateDoubleMatrix(cols, rows, mxComplexity::mxREAL);
	if (pa2 == NULL) {
		printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
		printf("Unable to create mxArray.\n");
		return(EXIT_FAILURE);
	}

	const int _size_pdata = rows*cols;

	double *pdata;
	pdata = (double*)mxGetPr(pa2);

	int k=0;
	for (int i=0; i< rows; i++)
	{
		for (int j=0; j< cols; j++)
		{
			//pdata[k] = (double)data_m.at<float>(i, j);
			pdata[k] = data_m.at<double>(i, j);
			//cout << pdata[k] << "\t";
			k++;
		}
	}

	//const char *file = "mattest.mat";
	int status; 

	printf("Creating file %s...\n\n", filename.c_str());
	pmat = matOpen(filename.c_str(), "w");
	if (pmat == NULL) {
		printf("Error creating file %s\n", filename.c_str());
		printf("(Do you have write permission in this directory?)\n");
		return(EXIT_FAILURE);
	}

	status = matPutVariable(pmat, "nemo", pa2);

	if (status != 0) {
		printf("Error using matPutVariableAsGlobal\n");
		return(EXIT_FAILURE);
	} 

	mxDestroyArray(pa2);


	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n",filename.c_str());
		return(EXIT_FAILURE);
	}

	printf("Done\n");
	//delete pdata;
	return(EXIT_SUCCESS);
}

cv::Mat ImageManager::DrawGPSHistogram(cv::Mat d_mat, string map_gps, float *alt, float *lng, int *max_voting)
{
	CV_Assert(d_mat.data != 0);
	CV_Assert(d_mat.channels() == 2);
	cout << d_mat.row(1) << endl;


	FileManager fm;
	cv::Mat gps_mat;
	fm.GpsDataReader(map_gps, &gps_mat);
	CV_Assert(gps_mat.data != 0);
	cout << gps_mat.row(0) << endl;
	cout << gps_mat.row(gps_mat.rows-1) << endl;

	cv::Mat hist_data;
	vector<cv::Mat> gps_data;
	double alt_min, alt_max, lng_min, lng_max;
	vector<cv::Mat> v_gps;
	cv::split(d_mat, v_gps);

	cv::Mat alt_col = v_gps[1];
	cv::Mat lng_col = v_gps[0];
	//cv::minMaxLoc(alt_col, &alt_min, &alt_max);
	//cv::minMaxLoc(lng_col, &lng_min, &lng_max);
	cv::minMaxLoc(gps_mat.col(1), &alt_min, &alt_max);
	cv::minMaxLoc(gps_mat.col(2), &lng_min, &lng_max);

	int alt_bins = gps_mat.rows;//hist_data.rows;//
	int lng_bins = gps_mat.rows;//hist_data.rows;//	

	gps_data.push_back(alt_col);
	gps_data.push_back(lng_col);
	cv::merge(gps_data, hist_data);
	hist_data = hist_data;
	hist_data.convertTo(hist_data, CV_32FC2);
	//cout << hist_data.row(0) << endl;

	//float *alt_ranges = (float *)malloc(sizeof(float) * alt_bins);// = {alt_min, alt_max};
	//float *lng_ranges = (float *)malloc(sizeof(float) * lng_bins);

	float alt_ranges[] = {alt_min, alt_max};
	float lng_ranges[] = {lng_min, lng_max};

	/*for (int i=0; i< alt_bins; i++)
	{
		alt_ranges[i] = alt_min +
	}*/

	//float lng_ranges[] = {lng_min, lng_max};
	float alt_range = alt_ranges[1] - alt_ranges[0];
	float lng_range = lng_ranges[1] - lng_ranges[0];

	
	//int alt_bins = ceil(alt_range/bin_size) , lng_bins = ceil(lng_range/bin_size);//1255;
	float bin_size_alt = alt_range / alt_bins;
	float bin_size_lng = lng_range / lng_bins;

	int histSize[] = {alt_bins, lng_bins};

	const float* ranges[] = {alt_ranges, lng_ranges};
	
	cv::MatND hist;
	int channels[] = {0, 1};
	
	cv::calcHist(&hist_data, 1,  channels, cv::Mat(), hist, 2, histSize, ranges, true, false);

	//cv::GaussianBlur(hist, hist, cv::Size(3,3), 0.5);

	double maxVal = 0;
	cv::Point minLoc, maxLoc;
	cv::minMaxLoc(hist, 0, &maxVal, &minLoc, &maxLoc);
	//cout << hist << endl;

	float alt_loc = bin_size_alt*maxLoc.y + alt_min;
	float lng_loc = bin_size_lng*maxLoc.x + lng_min;


	printf("alt=%3.8f, lng=%3.8f, maxVal=%3.8f", alt_loc, lng_loc, maxVal);
	
	(*alt) = alt_loc;
	(*lng) = lng_loc;
	(*max_voting) = maxVal;
	return hist;
}

cv::Mat ImageManager::MatchPruned(cv::Mat d_mat, int D)
{
	CV_Assert(d_mat.data != 0);
	vector<double> voting;

	for (int i=0; i<d_mat.rows; i++)
	{
		cv::Mat row = d_mat.row(i);
		//cout << row.type() << endl;
		int min_id = INT_MAX;
		double Loc_NND_1_alt = row.at<double>(0, 2);
		double Loc_NND_1_lng = row.at<double>(0, 3);
		for (int j=1; j < row.cols/4; j++)
		{
			double Loc_NND_j_alt = row.at<double>(0, j*4+2);
			double Loc_NND_j_lng = row.at<double>(0, j*4+3);
			double _D = sqrt(pow((Loc_NND_j_alt - Loc_NND_1_alt),2.0)+pow((Loc_NND_j_lng - Loc_NND_1_lng),2.0));
			if (_D < D && j < min_id)
			{
				min_id = j;
			}
		}
		if (min_id != INT_MAX)
		{
			double Dist_NND_1 = row.at<double>(0, 1);
			double Dist_NND_j = row.at<double>(0, min_id*4 + 1);
			double ratio = Dist_NND_1 / Dist_NND_j;
			if (ratio > 0.8)
			{
				voting.push_back(1);
			}
			else
			{
				voting.push_back(0);
			}
		}
		else
		{
			voting.push_back(0);
		}
	}

	cv::Mat NND_1_lat = d_mat.col(2);
	cv::Mat NND_1_alt = d_mat.col(3);
	cv::Mat vote(voting);
	//PointProductBtwMats<double>(NND_1_alt, vote, &NND_1_alt);
	//PointProductBtwMats<double>(NND_1_lat, vote, &NND_1_lat);
	vector<cv::Mat> gps;
	gps.push_back(NND_1_alt);
	gps.push_back(NND_1_lat);
	cv::Mat gps_data;
	cv::merge(gps, gps_data);
	return gps_data;
}

/*
Using k-means to do cluster, each cluster center is recognized as a visual word
*/
cv::Mat ImageManager::BoWExtractor(cv::Mat d_mat, int centers /* = 1000 */)
{
	CV_Assert(d_mat.data != 0);
	cv::BOWKMeansTrainer trainer(centers); //using k-means to get visual word
	trainer.add(d_mat);
	cv::Mat vocabulary = trainer.cluster();
	return vocabulary;
}

//template<class T>
//void ImageManager::PointProductBtwMats(cv::Mat imat1, cv::Mat imat2, cv::Mat *o_mat)
//{
//	CV_Assert(imat1.type() == imat2.type());
//	CV_Assert(imat1.size == imat2.size);
//
//	if (!o_mat->data)
//	{
//		o_mat->create(imat1.rows, imat1.cols, imat1.type());
//	}
//
//	for (int i=0; i < imat1.rows; i++)
//	{
//		for (int j=0; j<imat1.cols; j++)
//		{
//			(*o_mat).at<T>(i, j) = imat1.at<T>(i, j) * imat2.at<T>(i, j);
//		}
//	}
//}

void ImageManager::MSERFeatureDetector(cv::Mat imat)
{
	CV_Assert(imat.data != 0);
	
	cv::Mat gray_mat, ellipses;
	cv::cvtColor(imat, gray_mat, cv::COLOR_BGR2GRAY);
	imat.copyTo(ellipses);

	vector<vector<cv::Point>> contours;
	double t = (double)cv::getTickCount();
	cv::MSER()(gray_mat, contours);
	t = (double)cv::getTickCount() - t;
	printf("MSER extracted %d contours in %g ms. \n", (int)contours.size(), t * 1000./cv::getTickCount());

	vector<cv::Mat> v_patches;

	for (int i=(int)contours.size()-1; i>=0;i--)
	{
		const vector<cv::Point> &r = contours[i];
		cv::Mat m_r(r);
		vector<cv::Mat> v_points;
		cv::split(m_r, v_points);

		double maxv, minv;
		cv::minMaxLoc(v_points[0], &maxv, &minv);

		for (int j=0; j<(int)r.size(); j++)
		{
			cv::Point pt = r[j];
			
			imat.at<cv::Vec3b>(pt) = bcolors[i%9];			
		}

		cv::RotatedRect box = cv::fitEllipse(r);
		box.angle = (float)CV_PI/2-box.angle;
		cv::ellipse(ellipses, box, cv::Scalar(196,255,255), 2);
	}

	cv::imshow("Original", imat);
	cv::imshow("response", ellipses);
	cv::waitKey(0);
}

cv::Mat ImageManager::AffineTransform(cv::Mat inputMat)
{
	CV_Assert(inputMat.data != 0); 

	cv::Point2f srcTri[3];
	cv::Point2f dstTri[3];

	cv::Mat warp_mat(2, 3, CV_32FC1);
	cv::Mat warp_dst, warp_rotate_dst;

	//set the dst image the same type and size as input Mat
	warp_dst = cv::Mat::zeros(inputMat.rows, inputMat.cols, inputMat.type());
	
	//set 3 points to calculate the affine transform
	srcTri[0] = cv::Point2f(0, 0);
	srcTri[1] = cv::Point2f(inputMat.cols - 1, 0);
	srcTri[2] = cv::Point2f(0, inputMat.rows - 1);

	/*dstTri[0] = cv::Point2f(inputMat.cols * 0.0,	inputMat.rows * 0.33);
	dstTri[1] = cv::Point2f(inputMat.cols * 0.85,	inputMat.rows * 0.25);
	dstTri[2] = cv::Point2f(inputMat.cols * 0.15,	inputMat.rows * 0.7);*/
	dstTri[0] = srcTri[0];
	dstTri[1] = cv::Point2f(inputMat.cols * 0.85,	inputMat.rows * 0.25);
	dstTri[2] = srcTri[2];

	//Get affine transform
	warp_mat = getAffineTransform(srcTri, dstTri);

	//Apply the affine transform just found to the src image
	cv::warpAffine(inputMat, warp_dst, warp_mat, warp_dst.size());

#if _DEBUG
	cv::RNG rng(12345);
	
	cv::circle(warp_dst, dstTri[0], 5, cv::Scalar::all(-1), 3);
	cv::circle(warp_dst, dstTri[1], 5, cv::Scalar::all(-1), 3);
	cv::circle(warp_dst, dstTri[2], 5, cv::Scalar::all(-1), 3);
	
#endif

	/*cv::Point2f start = cv::Point2f(dstTri[2].x, dstTri[0].y);
	float height = dstTri[2].y - start.y - (start.y - dstTri[1].y);
	float width  = dstTri[1].x - start.x;*/
	cv::Point2f start = cv::Point2f(dstTri[0].x, dstTri[1].y);
	float height = dstTri[2].y - start.y;
	float width  = dstTri[1].x - start.x;
#if _DEBUG
	cv::circle(warp_dst, start, 5, cv::Scalar(-1), 3);
	cv::rectangle(warp_dst, cv::Rect(start.x, start.y, width, height), cv::Scalar::all(-1), 3);
	//cv::resize(warp_dst, warp_dst, cv::Size(warp_dst.cols/2, warp_dst.rows/2));
	//cv::imshow("Affine Transform", warp_dst);
	//cv::waitKey(0);
#endif
	return warp_dst(cv::Rect(start.x, start.y, width, height));
}

cv::Mat ImageManager::AffineTransform2(cv::Mat inputMat)
{
	CV_Assert(inputMat.data != 0); 

	cv::Point2f srcTri[3];
	cv::Point2f dstTri[3];

	cv::Mat warp_mat(2, 3, CV_32FC1);
	cv::Mat warp_dst, warp_rotate_dst;

	//set the dst image the same type and size as input Mat
	warp_dst = cv::Mat::zeros(inputMat.rows, inputMat.cols, inputMat.type());
	
	//set 3 points to calculate the affine transform
	srcTri[0] = cv::Point2f(0, 0);
	srcTri[1] = cv::Point2f(inputMat.cols - 1, 0);
	srcTri[2] = cv::Point2f(0, inputMat.rows - 1);

	dstTri[0] = cv::Point2f(inputMat.cols * 0.0,	inputMat.rows * 0.33);
	dstTri[1] = cv::Point2f(inputMat.cols * 0.85,	inputMat.rows * 0.25);
	dstTri[2] = cv::Point2f(inputMat.cols * 0.15,	inputMat.rows * 0.7);
	
	//Get affine transform
	warp_mat = getAffineTransform(srcTri, dstTri);

	//Apply the affine transform just found to the src image
	cv::warpAffine(inputMat, warp_dst, warp_mat, warp_dst.size());

#if _DEBUG
	cv::RNG rng(12345);
	
	cv::circle(warp_dst, dstTri[0], 5, cv::Scalar::all(-1), 3);
	cv::circle(warp_dst, dstTri[1], 5, cv::Scalar::all(-1), 3);
	cv::circle(warp_dst, dstTri[2], 5, cv::Scalar::all(-1), 3);
	
#endif

	cv::Point2f start = cv::Point2f(dstTri[2].x, dstTri[0].y);
	float height = dstTri[2].y - start.y - (start.y - dstTri[1].y);
	float width  = dstTri[1].x - start.x;
	
#if _DEBUG
	cv::circle(warp_dst, start, 5, cv::Scalar(-1), 3);
	cv::rectangle(warp_dst, cv::Rect(start.x, start.y, width, height), cv::Scalar::all(-1), 3);
	//cv::resize(warp_dst, warp_dst, cv::Size(warp_dst.cols/2, warp_dst.rows/2));
	//cv::imshow("Affine Transform", warp_dst);
	//cv::waitKey(0);
#endif
	return warp_dst(cv::Rect(start.x, start.y, width, height));
}

cv::Mat ImageManager::RotationTransform(cv::Mat inputMat)
{
	CV_Assert(inputMat.data != 0);
	//Rotation the image
	cv::Mat rot_mat(2, 3, CV_32FC1);
	cv::Point center = cv::Point(inputMat.cols/2, inputMat.rows/2);
	double angle = 180.0;
	double scale = 1.0;

	//Get the rotation matrix with the specifications above
	rot_mat = cv::getRotationMatrix2D(center, angle, scale);

	// Rotate the warped image
	cv::Mat rot_dst;
	cv::warpAffine(inputMat, rot_dst, rot_mat, inputMat.size());

	return rot_dst;
}

//----Clip panaromio image distortion on the top and bottom
cv::Mat ImageManager::PanaImageClipper(cv::Mat imat)
{
	CV_Assert(imat.data != 0); 

	cv::Mat result;
	int x = 0;
	int y = imat.rows / 6;
	int width = imat.cols;
	int height = 3 * imat.rows/4 - y;

	//cv::Rect(x, y, width, height);
	result = imat(cv::Rect(x, y, width, height));
	return result;
}