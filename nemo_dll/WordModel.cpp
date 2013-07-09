#include "WordModel.h"
#include "FileManager.h"
#include "ImageManager.h"

using namespace nemo;

/*!
	\fn generateWordDictionary
Params:
	@param1: cv::Mat allFeatureSamples
			all feature vector which will be clustered by hierarchy kmeans
	@param2: int numberofClusters (default value is 1000)
			Number of clusters will be generated
return:
	cv::Mat words
		center of each cluster
description:
	Doing hierarchy kmean clusters on feature dataset
*/
cv::Mat WordModel::generateWordDictionary(cv::Mat allFeaturesSamples, int numberOfClusters)
{
    cvflann::KMeansIndexParams kmParams;
    cv::Mat words = cv::Mat::zeros(numberOfClusters, allFeaturesSamples.cols, CV_32F);
    cv::flann::hierarchicalClustering<cv::flann::L1<float>>(allFeaturesSamples, words, kmParams);
    return words;
}

/*!
	\fn generateFlannIndex
Params:
	@param1: cv::Mat clusterSamples
			feature vector which is needed to be indexed
	@param2: int km_branches
			kmean clusters parameters with default value of 32//64
	@param3: int km_it
			kmean clusters parameters with default value 11//5
return 
	cv::flann::Index flannIndex
description:
	Build Flann index over sample dataset
*/
cv::flann::Index WordModel::generateFlannIndex(cv::Mat clusterSamples, int km_branches/* =64 */, int km_it/* =5 */)
{
    cv::flann::KMeansIndexParams param(km_branches, km_it);
    cv::flann::Index flannIndex(clusterSamples, param);	
    return flannIndex;
}

/*!
	\fn generateClusterMembershipForFeatures
Params:
	@param1: cv::flann::Index flann_index
			FLANN Index trained before
	@param2: cv::Mat featureVector
			new feature vector needed to find nearest neighbor points
	@param3: int k (default value is 1)
			number of nearest neighbor of each new feature vector
return 
	cv::Mat indices
		nearest neighbor index
description:
	Find the nearest neighbor index of each new feature vector
*/
cv::Mat WordModel::generateClusterMembershipForFeatures(cv::flann::Index flann_index, cv::Mat featureVector, int k/* =1 */)
{
	cv::Mat indices = cv::Mat::zeros(featureVector.rows, k, CV_32S);
	cv::Mat distances = cv::Mat::zeros(featureVector.rows, k, CV_32F);

	int numbreOfCheckers = 100;

	flann_index.knnSearch(featureVector, indices, distances, k);
	return indices;
}

template <class T>
void WordModel::saveImgClusterToTrnFile(string filename, vector<cv::Mat> docs)
{
    CV_Assert(docs.size() > 0);
    FileManager fm;

    ofstream oFile(filename.c_str(), ios::out);
    oFile << docs.size() << "\n";

    for (int i=0; i< docs.size(); i++)
    {
        cv::Mat doc = docs[i];
        for (int j=0; j< doc.rows; j++)
        {
            oFile << doc.at<T>(0, j) << " ";
        }
        oFile << "\n";
    }
    oFile.close();	
}

//For training images, each images will be clipped by 1/3 part.
vector<cv::Mat> WordModel::imgTranslator(string imgFolder, string docFilename, string wordFolder)
{
    FileManager fm;
    ImageManager im;
    vector<string> v_wd = fm.FileReader(wordFolder, "\\*.wd");
    cv::Mat words;
    cv::flann::KMeansIndexParams params(64);
    cv::flann::Index index; //= generateFlannIndex(words);

    if(v_wd.size() == 0)
    {
        CubeMat cm = fm.CubeMatSampling(imgFolder, 5);
        cv::Mat allFeatureSamples;

        for (int i=0; i<cm.size(); i++)
        {
            vector<string> cube = cm[i];
            for (int j=0; j<cube.size(); j++)
            {
                cv::Mat imat = cv::imread(cube[++j]);
                imat = im.ImageClipper(imat);
                cv::Mat descriptor = im.ImgSiftCollector(imat, false, 0.9);
                allFeatureSamples.push_back(descriptor);
            }
            cout << "\rCube_" << i << "is done.";
        }

        fm.SaveMat2Disk<float>(allFeatureSamples, "model\\feature.sift");

        words = generateWordDictionary(allFeatureSamples, 200000);
        fm.SaveMat2Disk<float>(words, "model\\word.wd"); 

        index.build(words, params);
        index.save("model\\index.flann");
    }
    else
    {		
        for (int i=0; i < v_wd.size(); i++)
        {
            string wd_file = v_wd[i];
            cv::Mat _words;
            fm.ReadMatFromDisk<float>(wd_file, &_words);
            words.push_back(_words);
        }		

        index.load(words, wordFolder+"\\index.flann");
    }

    //Transfer image set into GibbsLDA++ document
    vector<cv::Mat> v_docs;
    CubeMat cm = fm.CubeMatSampling(imgFolder, 5);
    //CubeMat cm = fm.CubeMatBuilder(imgFolder);
    cv::Mat allFeatureSamples;
    for (int i=0; i < cm.size(); i++)
    {
        vector<string> cube = cm[i];
        for (int j=0; j < cube.size(); j++)
        {
            cv::Mat imat = cv::imread(cube[++j]);
            imat = im.ImageClipper(imat);
            cv::Mat descriptor = im.ImgSiftCollector(imat, false, 0.9);
            //cv::Mat centers = generateClusterMembershipForFeatures(index, descriptor, 1);
            cv::Mat distance, indices;
            index.knnSearch(descriptor, indices, distance, 1);
            v_docs.push_back(indices);
        }
        cout << "\rCube_" << i << "is done.";
    }

    saveImgClusterToTrnFile<unsigned>("model\\trndocs.data", v_docs);
    return v_docs;
}

/*!
	\fn clusterImageBasedonTopicDist
Params:
	@param1: string thetaFilename
return:
	cv::Mat center
	center topic distribution
cluster all image topic distribution into 20 groups by kmeans, the center topic distribution is returned
*/
cv::Mat WordModel::clusterImageBasedonTopicDist(string thetaFilename, string imgfolder)
{
	cv::Mat thetaMat;
	FileManager fm;
	ImageManager im;
	fm.ThetaDataReader(thetaFilename, &thetaMat);
	//fm.ReadMatFromDisk<float>(thetaFilename, &thetaMat);
	CV_Assert(thetaMat.data != 0);

	cv::Mat labels, centers;
	cv::kmeans(thetaMat, 32, labels, cv::TermCriteria(), 5, 0, centers);
	//fm.SaveMat2Disk<float>(centers, "model\\flann\\center.theta");

	vector<vector<int>> v_groups(32);
	for (int i=0; i<thetaMat.rows; i++)
	{
		int idx = labels.at<int>(i, 0);
		v_groups[idx].push_back(i);
	}

	cv::Mat gps_mat;
	fm.ReadMatFromDisk<double>("model\\gps.dat", &gps_mat);
	CubeMat cm = fm.CubeMatSampling(imgfolder, 5);

	//organize sampled images into one vector, only double side views will be inserted
	vector<string> imglist;
	for (int i=0; i<cm.size(); i++)
	{
		imglist.push_back(cm[i][1]);
		imglist.push_back(cm[i][3]);
	}

#if 0
	for (int i=0; i<v_groups.size(); i++)
	{
		vector<int> _v = v_groups[i];
		cv::Mat sift_features, gps_features;
		for (int j=0; j < _v.size(); j++)
		{
			int idx = _v[j];
			double lat = gps_mat.at<double>(idx, 0);
			double lng = gps_mat.at<double>(idx, 1);
			cv::Mat gps_row(1,2, CV_64F);
			gps_row.at<double>(0,0) = lat;
			gps_row.at<double>(0,1) = lng;

			string imgname = imglist[idx];

			cv::Mat imat = cv::imread(imgname);
			imat = im.ImageClipper(imat);
			cv::Mat descriptor;
			descriptor = im.ImgSiftCollector(imat, false, 0.9);
			sift_features.push_back(descriptor);
			for (int k=0; k<descriptor.rows; k++)
			{
				gps_features.push_back(gps_row);
			}			
		}
		char buffer[256];
		sprintf(buffer, "model\\flann\\%d.sift", i);
		fm.SaveMat2Disk<float>(sift_features, static_cast<string>(buffer));
		sprintf(buffer, "model\\flann\\%d.gps", i);
		fm.SaveMat2Disk<double>(gps_features, static_cast<string>(buffer));
	}
#else
	string trainingimgclusterfolder = "imgs\\pitts";

	for (int i=0; i<v_groups.size(); i++)
	{
		vector<int> _v=v_groups[i];
		for (int j=0; j<_v.size(); j++)
		{
			int idx = _v[j];
			string imgname = imglist[idx];
			char buffer[256];
			sprintf(buffer, "%s\\%d\\", trainingimgclusterfolder.c_str(), i);
			string dstfolder = fm.DirectoryBuilder(static_cast<string>(buffer));
			
			vector<string> _v;
			fm.file_name_splitter(imgname, &_v, "\\");
			dstfolder = dstfolder + _v[_v.size() - 1];

			fm.copyFile(imgname, dstfolder);
		}
	}
#endif

	/*
	cv::flann::Index index(centers, cv::flann::KMeansIndexParams());

	cv::Mat thetaMat_test, indices, distance;
	string thetafilename_test = "model\\test_docs.dat.theta";
	fm.ThetaDataReader(thetafilename_test, &thetaMat_test);
	index.knnSearch(thetaMat_test, indices, distance, 1);
	
	for (int i=0; i < indices.rows; i++)
	{
		cout << indices.row(i) << endl;
	}
	getchar();
	*/
	return centers;
}