#ifndef _NEMO_COMMON_
#define _NEMO_COMMON_

#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/flann/miniflann.hpp>
#include <opencv2/flann/flann.hpp>
#include <opencv2/flann/hierarchical_clustering_index.h>
#include <opencv2/core/core.hpp>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/constants.hpp>
#include <boost/filesystem.hpp>
#include <boost/regex.hpp>
#include <boost/tuple/tuple.hpp>
#include <boost/multi_index/ordered_index.hpp>
#include <boost/multi_index_container.hpp>
#include <boost/multi_index/member.hpp>
#include <boost/multi_index/composite_key.hpp>
#include <boost/next_prior.hpp>
#include <boost/tokenizer.hpp>
#include <boost/program_options.hpp>
#include <boost/thread/thread.hpp>

#include <mat.h>
#include <algorithm>
#include <vector>
#include <string>
#include <map>
#include <time.h>
#include <iostream>
#include <fstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <direct.h>
#include <io.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>


#include <log4cplus/appender.h>
#include <log4cplus/logger.h>
#include <log4cplus/fileappender.h>
#include <log4cplus/layout.h>
#include <log4cplus/ndc.h>
#include <log4cplus/helpers/loglog.h>
#include <log4cplus/loggingmacros.h>
#include <log4cplus/helpers/stringhelper.h>

#include <gsl/gsl_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_statistics.h>

using namespace std;
namespace bf = boost::filesystem;
namespace bt = boost::tuples;
namespace bm = boost::multi_index;
namespace bpo = boost::program_options;

namespace nemo
{
	//--typedef for GibbsLDA++
	typedef map<string, int> mapword2id;
	typedef map<int, string> mapid2word;

	enum Transformation_Type
	{
		Rotation = 0,
		Affine   = 1,
		Affine2	 = 2
	};

	//------------------------
	typedef	vector<vector<string>> CubeMat;

#define NEMO_IMAGE_NAME_REG				"reprojected_([0-9]+)_([0-9]+)"
#define NEMO_KNN_NEIG 5
#define NEMO_SIFT_THRESHOLD 1.6

#define NEMO_KB 1024
#define NEMO_MB NEMO_KB * 1024
#define NEMO_GB NEMO_MB * 1024

	//Constant for GibbsLDA++
#define	NEMO_BUFF_SIZE_LONG	1000000
#define NEMO_TESTIMG_THETA_FILENAME "testImg.dat.theta"
#define NEMO_TESTIMG_DATA_FILENAME "testImg.dat"
	//------------------------------

	enum FOLDERSTATE
	{
		_IS_FOLDER = 0,
		_IS_FILE = 1,
		_NOT_EXISTED = 2
	};

	struct TstImg 
	{
		string filename;
		double lat;
		double lng;

		TstImg(){}

		TstImg(string _name, double _lat, double _lng):
			filename(_name), lat(_lat), lng(_lng){}

		TstImg &operator=(const TstImg &ti)
		{
			filename = ti.filename;
			lat = ti.lat;
			lng = ti.lng;

			return *this;
		}
	};

	struct ImgCube 
	{
		string f_view;
		string l_view;
		string b_view;
		string r_view;

		ImgCube(){}

		ImgCube(string _f, string _l, string _b, string _r):
			f_view(_f), l_view(_l), b_view(_b), r_view(_r){}

		ImgCube &operator=(const ImgCube &ic)
		{
			f_view = ic.f_view;
			l_view = ic.l_view;
			b_view = ic.b_view;
			r_view = ic.r_view;
			return *this;
		}
	};

	struct NNPoint
	{
		int id;
		int indics;
		double dist;
		double alt;
		double lng;

		NNPoint(){}

		NNPoint(int _id, int _indics, double _dist, double _alt=0, double _lng=0): id(_id), indics(_indics), dist(_dist), alt(_alt), lng(_lng){}
		bool operator< (const NNPoint &nnp) const 
		{
			//return dist < nnp.dist;
			return id < nnp.id;
		}
	};

	struct  GPS_location
	{
		int id;
		double alt;
		double lng;
		int voting;

		GPS_location(){}
		GPS_location(int _id, double _alt, double _lng, int _voting):id(_id), alt(_alt), lng(_lng), voting(_voting){}
		bool operator<(const GPS_location &gpsl) const
		{
			return id < gpsl.id;
			//return voting < gpsl.voting;
		}
	};

	struct SiftCube
	{
		int id;
		int view_size;
		vector<cv::Mat> _v_descriptors;

		SiftCube(){}
		SiftCube(int _view_size){
			view_size = _view_size;
			_v_descriptors.resize(_view_size);
		}

		SiftCube &operator=(const SiftCube &sc)
		{
			id = sc.id;
			view_size = sc.view_size;
			_v_descriptors = sc._v_descriptors;
			return *this;
		}
	};

	struct SiftGPS 
	{
		cv::Mat featureVector;
		cv::Mat gpsFeatureVector;

		SiftGPS &operator=(const SiftGPS &sg)
		{
			featureVector		= sg.featureVector;
			gpsFeatureVector	= sg.gpsFeatureVector;
			return *this;
		}
	};



	struct _GPS_location
	{
		double lat;
		double lng;

		_GPS_location(){};

		_GPS_location(double _lat, double _lng):lat(_lat), lng(_lng){};
	};

	struct FeatureParams
	{
		string imgFolder;
		string gpsFilename;
		string outputFolder;
		bool isAppend;
		string featureType;
		double siftBlurCof;
		float fileSize;
		int sampling;

		FeatureParams(){}

		FeatureParams(string _imgFolder, string _gpsFilename, string _outputFolder, bool _isAppend, string _featureType, double _siftBlurCof, float _fileSize)
			:imgFolder(_imgFolder), gpsFilename(_gpsFilename), outputFolder(_outputFolder), isAppend(_isAppend), featureType(_featureType), siftBlurCof(_siftBlurCof), fileSize(_fileSize){}
	};

	struct WordModelParams
	{
		string imgFolder;
		string gpsFilename;
		string featureFolder;
		string outputFolder;
		string wordFolder;
		string flannFolder;
		string docsFolder;
		string testImgFolder;
		float wordAmnt;
		float filesize;
		float accuracy_dist;


		WordModelParams(){}

		WordModelParams(string _featureFolder, float _wordAmnt, string _outputFolder, string _flannFilename, float _filesize=1.5):
			featureFolder(_featureFolder), wordAmnt(_wordAmnt), outputFolder(_outputFolder), flannFolder(_flannFilename), filesize(_filesize){};	
	};

	struct searchParams
	{
		string featureFolder;
		string wordFolder;
		string outputFolder;
		string flannFilename;
		string wordFilename;
		string wordEntropyFilename;
		string wordGpsFilename;
		string gpsFilename;
		float siftBlurCof;
		string testImgFolder;
		string sourceImgFolder;
		int testImgAmt;
		bool isShow;
	};

	typedef bm::multi_index_container<
		NNPoint, 
		bm::indexed_by<
		bm::ordered_unique<boost::multi_index::identity<NNPoint>>,
		bm::ordered_non_unique<bm::member<NNPoint, double, &NNPoint::dist>>	
		>
		> NNP_Set;		

	typedef bm::multi_index_container<
		GPS_location,
		bm::indexed_by<
		bm::ordered_unique<boost::multi_index::identity<GPS_location>>,
		bm::ordered_non_unique<bm::member<GPS_location, int, &GPS_location::voting>>
		>
		> GPSL_Set;

	//--------for word distribution----------------//
	struct word
	{
		int id;
		double wid;
		double dist;
		double lat; 
		double lng;

		word(){};

		word(int _id, double _wid, double _dist, double _lat, double _lng):id(_id), wid(_wid), dist(_dist), lat(_lat), lng(_lng){}

		bool operator<(const word &_wd) const
		{
			return id < _wd.id;
			//return voting < gpsl.voting;
		}
	};


	typedef bm::multi_index_container<
		word,
		bm::indexed_by<
		bm::ordered_unique<boost::multi_index::identity<word>>,
		bm::ordered_non_unique<bm::member<word, double, &word::wid>>,
		bm::ordered_non_unique<bm::member<word, double, &word::lat>>
		>
		> Word_Set;

	// GPS structure for GPS index by lat and lng [11/3/2012 100464067]
	struct GPS_Tuple
	{
		double id;
		double lat;
		double lng;
		double voting;

		GPS_Tuple(){};

		GPS_Tuple(double _id, double _lat, double _lng, double _voting=0):
			id(_id), lat(_lat), lng(_lng), voting(_voting){};
	};

	struct gps_key:bm::composite_key <
		GPS_Tuple,
		BOOST_MULTI_INDEX_MEMBER(GPS_Tuple, double, lat),
		BOOST_MULTI_INDEX_MEMBER(GPS_Tuple, double, lng)
	>{};

	typedef bm::multi_index_container<
		GPS_Tuple,
		bm::indexed_by<
		bm::ordered_unique<
		gps_key, bm::composite_key_result_less<gps_key::result_type>
		>
		>> GPS_Set;

	typedef bm::nth_index<GPS_Set, 0>::type GPS_by_latlng;
	//---------------------------------------------------------------------
	struct testImgSift
	{
		string imgname;
		cv::Mat descriptor;
		cv::Mat words;
		cv::Mat dists;
		cv::Mat voting_tag;
		double glat;
		double glng;
		double lat;
		double lng;
		int result_id;
		vector<NNP_Set> vnnps;
		GPS_Set gs;
		int voting_amt;
		vector<float> groups;

		testImgSift &operator=(const testImgSift &tis)
		{
			imgname = tis.imgname;
			descriptor = tis.descriptor;
			words = tis.words;
			dists = tis.dists;
			voting_tag = tis.voting_tag;
			glat = tis.glat;
			glng = tis.glng;
			lat = tis.lat;
			lng = tis.lng;
			result_id = tis.result_id;
			vnnps = tis.vnnps;
			gs = tis.gs;
			voting_amt = tis.voting_amt;
			return *this;
		}
	};
	//----------The end of word distribution------------//

	struct SearchParams
	{
		string imgFolder;
		string gpsFilename;
		string siftFolder;
		string outputFolder;
		string testImgFolder;
		string wordFolder;
		string trGpFolder;
		string modelFolder;
		string modelName;
		string amirFolder;

		bool isBuildFlann;
		float siftBlurCoef;
		int testAmt;
		bool isSameDB;
		bool isPruned;
		bool isPana; // Panoramic images
		bool isTransf;
		double pruneD;
		double accuracy_dist;
		Transformation_Type tt;
	};

	struct GroupParams
	{
		string imgFolder;
		string thetaFilename;                
		string outputFolder;
        string fuzzyMatlabfilename;
        string fuzzyMatlabClusterCenterfilename;

		int step;
		int group;
		double fuzzyRatio;
	};
}
#endif