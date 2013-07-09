#include "GroupModel.h"
#include "nemo_dll\FileManager.h"
#include "nemo_dll\WordModel.h"

using namespace nemo;

void GroupModel::GroupingDBFrTheta(GroupParams gp)
{
    FileManager fm;
    CubeMat cm = fm.CubeMatSampling(gp.imgFolder, gp.step);
    CV_Assert(cm.size() > 0);

    std::cout << "Read Theta file..." << std::endl;

    string thetaFilename = gp.thetaFilename;
    cv::Mat thetaMat;
    fm.ThetaDataReader(thetaFilename, &thetaMat);
    int group = gp.group;

    std::cout << "Doing kmean on thetaMat..." << std::endl;
    cv::Mat labels, centers;
    cv::kmeans(thetaMat, group, labels, cv::TermCriteria(), 5, 0, centers);

    vector<vector<int>> v_group(group);
    for (int i=0; i<thetaMat.rows; i++)
    {
        int idx = labels.at<int>(i, 0);
        v_group[idx].push_back(i);
    }

    // Read images into vector list
    int docAmt = cm.size() * cm[0].size();
    CV_Assert(docAmt == thetaMat.rows);
    std::cout << "Total " << docAmt << " images" << std::endl;
    vector<string> imgList;
    for (int i=0; i<cm.size(); i++)
    {
        for (int j=0; j<cm[i].size(); j++)
        {
            imgList.push_back(cm[i][j]);
        }
    }

    std::cout << "Start copy images accordint to groups..." << std::endl;
    // Grouping training images, copy images into group folder
    string trainingImgClusterFolder = gp.outputFolder;
    fm.DirectoryBuilder(trainingImgClusterFolder + "\\");
    fm.SaveMat2Disk<float>(centers, trainingImgClusterFolder + "\\centers.mat");
    WordModel wm;
    cv::flann::Index idx = wm.generateFlannIndex(centers, 25, 5);
    idx.save(trainingImgClusterFolder + "\\centers.flann");

    for (int i=0; i<v_group.size(); i++)
    {
        vector<int> _v = v_group[i];
        for (int j=0; j < _v.size(); j++)
        {
            int _idx = _v[j];
            string imgName = imgList[_idx];
            char buffer[256];
            string zero = fm.zeros(3, i);
            sprintf(buffer, "%s\\%s%d\\", trainingImgClusterFolder.c_str(), zero.c_str(), i);
            string dstFolder = fm.DirectoryBuilder(static_cast<string>(buffer));

            vector<string> _vt;
            fm.file_name_splitter(imgName, &_vt, "\\");
            dstFolder = dstFolder + _vt[_vt.size()-1];
            fm.copyFile(imgName, dstFolder);
        }
    }
}

void GroupModel::FuzzyGroupDBFrTheta(GroupParams gp)
{
    FileManager fm;

    // Read fuzzy cluster from matlab mat
    cv::Mat fzClusterMat;
    fm.ReadMatlabMatfrDisk(gp.fuzzyMatlabfilename, &fzClusterMat);

    // Read image list
    vector<string> imgList = ImageListGenerator(gp);
    CV_Assert(fzClusterMat.cols == imgList.size());
#if 0
    std::cout << fzClusterMat.col(0);   
#endif
    cv::Mat fuzzyCenter;
    fm.ReadMatlabMatfrDisk(gp.fuzzyMatlabClusterCenterfilename, &fuzzyCenter);
    fuzzyCenter.convertTo(fuzzyCenter, CV_32FC1);
#if _DEBUG
    std::cout << fuzzyCenter.col(0) << std::endl;
#endif

    CV_Assert(fzClusterMat.data != 0);
    CV_Assert(fuzzyCenter.data != 0);

    vector<vector<int>> v_groups(fzClusterMat.rows);
    for (int c=0; c < fzClusterMat.cols; c++) // Each column is a image
    {
        cv::Mat col = fzClusterMat.col(c);
        SoftGroup(col, c, &v_groups, gp.fuzzyRatio);
    }

    string trainingImgClusterFolder = gp.outputFolder;
    fm.DirectoryBuilder(trainingImgClusterFolder + "\\");
    fm.SaveMat2Disk<float>(fuzzyCenter, gp.outputFolder + "\\centers.mat");
    WordModel wm;
    
    //cv::flann::Index idx = wm.generateFlannIndex(fuzzyCenter, 5, 5);    
    cv::flann::AutotunedIndexParams params;
    cv::flann::Index idx(fuzzyCenter, params);
    idx.save(trainingImgClusterFolder + "\\centers.flann");

    std::cout << "Start copy images accordint to groups..." << std::endl;
    // Grouping training images, copy images into group folder
    
    CopyImg2Group(v_groups, trainingImgClusterFolder, imgList);

}

// Group image into cluster based on its probability
void GroupModel::SoftGroup(cv::Mat cols, int col_idx, vector<vector<int>> *v_groups, double fuzzyThreshold)
{
    CV_Assert(v_groups->size()!=0);
    CV_Assert(cols.cols==1); // Only column vector

    double max_prob, min_prob;
    cv::Point max_idx, min_idx;
    cv::minMaxLoc(cols, &min_prob, &max_prob, &min_idx, &max_idx, cv::Mat());
    (*v_groups)[max_idx.y].push_back(col_idx);

    for (int i=0; i<cols.rows; i++)
    {
        double prob = cols.at<double>(i, 0);

        if (prob/max_prob >= fuzzyThreshold && prob != max_prob)
        {
            (*v_groups)[i].push_back(col_idx);
        }
    }
}

vector<string> GroupModel::ImageListGenerator(GroupParams gp)
{
    vector<string> imgList;
    FileManager fm;
    CubeMat cm = fm.CubeMatSampling(gp.imgFolder, gp.step);
    CV_Assert(cm.size() > 0);
    for (int i=0; i<cm.size(); i++)
    {
        for (int j=0; j<cm[i].size(); j++)
        {
            imgList.push_back(cm[i][j]);
        }
    }

    return imgList;
}

void GroupModel::CopyImg2Group(vector<vector<int>> v_group, string outputFolder, vector<string> imgList)
{
    CV_Assert(v_group.size() > 0);
    CV_Assert(imgList.size() > 0);

    FileManager fm;

    for (int i=0; i<v_group.size(); i++)
    {
        vector<int> _v = v_group[i];
        for (int j=0; j < _v.size(); j++)
        {
            int _idx = _v[j];
            string imgName = imgList[_idx];
            char buffer[256];
            string zero = fm.zeros(3, i);
            sprintf(buffer, "%s\\%s%d\\", outputFolder.c_str(), zero.c_str(), i);
            string dstFolder = fm.DirectoryBuilder(static_cast<string>(buffer));

            vector<string> _vt;
            fm.file_name_splitter(imgName, &_vt, "\\");
            dstFolder = dstFolder + _vt[_vt.size()-1];
            fm.copyFile(imgName, dstFolder);
        }
    }
}