#pragma once
#include "nemo_dll\common.h"
#include "nemo_dll\FileManager.h"

namespace nemo
{

class WordUtilities
{
	struct voc_word
	{
		int wid;
		int freq;

		voc_word();
		voc_word(int _wid, int _freq):wid(_wid), freq(_freq){}
		bool operator< (const voc_word &_wd) const
		{
			return wid < _wd.wid;
		}
	};

	typedef bm::multi_index_container< 
		voc_word,
		bm::indexed_by<
		bm::ordered_unique<boost::multi_index::identity<voc_word> >,
		bm::ordered_unique<bm::member<voc_word, int, &voc_word::wid> >,
		bm::ordered_non_unique<bm::member<voc_word, int, &voc_word::freq> >
		> > Word_FREQ;

	struct img_word
	{
		int imgId;
		double freq;

		img_word();
		img_word(int _imgId, double _freq):imgId(_imgId), freq(_freq){}
		bool operator< (const img_word &_iw) const
		{
			return imgId < _iw.imgId;
		}
	};

	typedef bm::multi_index_container<
		img_word,
		bm::indexed_by<
		bm::ordered_unique<boost::multi_index::identity<img_word> >,
		bm::ordered_unique<bm::member<img_word, int, &img_word::imgId> >,
		bm::ordered_non_unique<bm::member<img_word, double, &img_word::freq> >
		> > Img_FREQ;

public:
	WordUtilities(void){};
	~WordUtilities(void){};

	void wordGeneration(string featureFolder, double clusterAmt);

	void word_entropy_builder(string ref_imgFolder, string wdFilename, string wdflannfilename);

	void invert_file(cv::Mat indics, int imgId, vector<vector<long int > > *word_invert);

	void invert_file_mat(cv::Mat indics, int imgId, cv::Mat *m_invert_file);

	void invert_file_analysis(string invertFilename, string gpsFilename);
	
	void findMinEnclosedCircle(vector<cv::Point2f> v_points, cv::Mat *center, float *radius);

	int vec_invert_filesize(vector<vector<long int > > *word_invert);

	vector<bt::tuple<long int, long int> > word_freq(cv::Mat _indics);

	void save_word_invert(vector<vector<long int > > word_invert, string filename);

	vector<vector<long int> > read_word_invert(string filename);

	//vector<cv::Mat> read_word_invert(string filename);

	void search_by_word(string wdfilename, string testImgFolder, string invert_filename, string gpsfilename, int type, string ref_imgFolder);

	vector<int> vote_by_word(cv::Mat indics, vector<vector<long int > > v_invert_file, cv::Mat radius_mat, int type);

	void save_result_images(vector<int> _v, string imgName);

	void generateModelTestImages(string ref_imgfolder, string gpsfilename, int imgAmt, string testImgFolder);
};

}
