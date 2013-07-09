#include "NemoUtilities.h"
#include "nemo_dll\WordModel.h"

using namespace nemo;

// Input folder contain all feature mats, gps mats and sidx files
void NemoUtilities::corpus2WDList(string corpusFolder, string wdFilename)
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

void NemoUtilities::corpus2TrnDatafile(string docFolder, string trnFilename)
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

void NemoUtilities::MatlabMatGenerator(string thetaFilename, string oMatfilename/* ="" */)
{
	FileManager fm;
	cv::Mat thetaMat;
	
	std::cout << "Reading ThetaMat..." << std::endl;
	fm.ThetaDataReader(thetaFilename, &thetaMat);
	thetaMat.convertTo(thetaMat, CV_64FC1);

	std::cout << "Save it to disk as " << oMatfilename << std::endl;
	fm.SaveMatlabMatfile2Disk(thetaMat, oMatfilename);
}

void NemoUtilities::wordGenerate(string featureFolder, double wdAmt)
{
    FileManager fm;
    WordModel wm;
    cv::Mat features;
    vector<string> featureList = fm.FileReader(featureFolder, "\\*.sift");
    CV_Assert(featureList.size()>0);

    for (int i=0; i<featureList.size(); i++)
    {
        cv::Mat features;
        fm.ReadMatFromDisk<float>(featureList[i], &features);
        CV_Assert(features.data != 0);

        int _wdAmt = features.rows*wdAmt;
        std::cout << _wdAmt << "words will be generating..." << std::endl;
        cv::Mat words = wm.generateWordDictionary(features, _wdAmt);
        string wdFilename = fm.FileExtensionChange(featureList[i], "doc");
        fm.SaveMat2Disk<float>(words, wdFilename);
    }
}

void NemoUtilities::DrawVisualWord(string imgListFolder, string wdfolder)
{
	FileManager fm;
	vector<string> imgList = fm.FileReader(imgListFolder, "\\*.jpg");
	CV_Assert(imgList.size() > 0); // if load image fail, stop!
	
	cv::Mat word;
	fm.ReadMatFromDisk<float>(wdfolder+"\\words.wd", &word);
	cv::flann::Index index;
	index.load(word, wdfolder + "\\words.flann");

	ImageManager im;
	cv::SiftFeatureDetector detector;
	cv::SiftDescriptorExtractor extractor;
	
	for (int i=0; i<imgList.size(); i++)
	{
		cv::Mat imat = cv::imread(imgList[i]);
		imat = im.ImageClipper(imat);
		cv::vector<cv::KeyPoint> keypoints;
		detector.detect(imat, keypoints);

		cv::Mat descriptor;
		extractor.compute(imat, keypoints, descriptor);

		CV_Assert(descriptor.data!=0);
		cv::Mat indics, dists;
		index.knnSearch(descriptor, indics, dists, 1);
		CV_Assert(indics.data != 0);
		indics.convertTo(indics, CV_32SC1);
		CV_Assert(indics.rows == keypoints.size());
		cv::Mat result = DrawColorSifts(indics, imat, keypoints);
		imwrite(imgList[i], result);
	}
}

cv::Mat NemoUtilities::DrawColorSifts(cv::Mat indics, cv::Mat image, vector<cv::KeyPoint> keypoints)
{
#define  PI 3.1415926
	for (int i=0; i < keypoints.size(); i++)
	{
		int index = indics.at<int>(i, 0);
		int r = index % 255;
		int g = (index / 255) % 255;
		int b = (index / (255 * 255)) % 255;

		cv::Scalar color(b, g, r);

		cv::KeyPoint kp = keypoints[i];
		cv::Point2f p1, p2, p3, p4, p5;
		p1.x = kp.pt.x - kp.size*cos(kp.angle * PI/180);
		p1.y = kp.pt.y + kp.size*sin(kp.angle * PI/180); 
		p2.x = kp.pt.x + kp.size*sin(kp.angle * PI/180);
		p2.y = kp.pt.y + kp.size*cos(kp.angle * PI/180);
		p3.x = kp.pt.x + kp.size*cos(kp.angle * PI/180);
		p3.y = kp.pt.y - kp.size*sin(kp.angle * PI/180);
		p4.x = kp.pt.x - kp.size*sin(kp.angle * PI/180);
		p4.y = kp.pt.y - kp.size*cos(kp.angle * PI/180);
		p5.x = kp.pt.x + (p1.x - p4.x)/2;
		p5.y = kp.pt.y + (p1.y - p4.y)/2;

		cv::line(image, p1, p2, color);
		cv::line(image, p2, p3, color);
		cv::line(image, p3, p4, color);
		cv::line(image, p4, p1, color);
		cv::line(image, kp.pt, p5, color);
	}


	cv::imshow("SIFT", image);
	cv::waitKey(0);

	return image;

}

