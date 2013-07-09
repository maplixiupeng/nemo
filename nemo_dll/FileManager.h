#pragma once
#include "common.h"
#include "ImageManager.h"

namespace nemo
{

    class FileManager
    {
    public:
        FileManager(void){};
        ~FileManager(void){};

        template <class T>
        static __declspec(dllexport)
            inline void SaveMat2Disk(cv::Mat mData, string filename)
        {
            ofstream myFile(filename.c_str(), ios::out|ios::binary);
            int rows = mData.rows;
            int cols = mData.cols;
            myFile.write((char*)&rows, sizeof(int));
            myFile.write((char*)&cols, sizeof(int));

            /*T *data = new T[rows*cols];
            int k=0;
            for (int i=0; i<mData.rows; i++)
            {
            for (int j=0; j<mData.cols;j++)
            {
            data[k] = mData.at<T>(i, j);
            k++;
            }
            }*/

            //myFile.write((char*)data, sizeof(T)*rows*cols);
            myFile.write((char*)mData.data, sizeof(T)*rows*cols);
            myFile.close();
        }

        template <class T>
        static __declspec(dllexport)
            void SaveMatVect2Disk(vector<cv::Mat> vMat, string filename)
        {
            if (vMat.size() <= 0) // null vector
            {
                return;
            }
            ofstream myfile(filename.c_str(), ios::out | ios::binary);
            int vec_length = vMat.size();
            myfile.write((char*)&vec_length, sizeof(int)); // vector length

            for (int i=0; i<vMat.size(); i++)
            {
                cv::Mat mat = vMat[i];
                int rows = mat.rows;
                int cols = mat.cols;
                myfile.write((char*)&rows, sizeof(int));
                myfile.write((char*)&cols, sizeof(int));
                myfile.write((char*)mat.data, sizeof(T)*rows*cols);
            }

            myfile.close();
        }

        template <class T>
        static __declspec(dllexport)
            inline void SaveBinaryGSLMat2Disk(T *arr, int rows, int cols, string filename)
        {
            ofstream myFile(filename.c_str(), ios::out|ios::binary);
            myFile.write((char*)&rows, sizeof(int));
            myFile.write((char*)&cols, sizeof(int));

            myFile.write((char*)arr, sizeof(T)*rows * cols);
            myFile.close();
        }

        template <class T>
        static __declspec(dllexport)
            inline T* ReadBinarytGSLFromDisk(string filename, int *rows, int *cols)
        {
            bf::path p(filename);
            if(bf::exists(p))
            {
                ifstream myFile(filename.c_str(), ios::in|ios::binary);
                int _rows; 
                int _cols;
                myFile.read((char*)&_rows, sizeof(int));
                myFile.read((char*)&_cols, sizeof(int));

                T* arr=new T[_rows*_cols];
                myFile.read((char*)arr, sizeof(T)*_rows*_cols);
                *rows = _rows;
                *cols = _cols;
                return arr;
            }
            else
            {
                cerr << filename << " doesn't exist" << endl;
                return 0;
            }
        }

        template <class T>
        static __declspec(dllexport)
            inline void ReadMatFromDisk(string filename, cv::Mat *mData)
        {
            bf::path p(filename);
            if (bf::exists(p))
            {
                ifstream myFile(filename.c_str(), ios::in|ios::binary);
                int rows;
                int cols;
                myFile.read((char*) &rows, sizeof(int));
                myFile.read((char*) &cols, sizeof(int));
                T *data = new T[rows*cols];
                myFile.read((char*) data, sizeof(T)*rows*cols);


                cv::Mat head(rows, cols, cv::DataType<T>::type, data);

                *mData = head;			

                /*mData->create(rows, cols, cv::DataType<T>::type);
                int k=0;
                for (int i=0; i<rows; i++)
                {
                for (int j=0;j<cols;j++)
                {
                mData->at<T>(i, j)=data[k];
                k++;
                }
                }*/
            }
            else
            {
                cerr << filename << " doesn't exist" << endl;
            }
        }

        template <class T>
        static __declspec(dllexport)
            void ReadMatVect2Disk(string filename, vector<cv::Mat> *vMat)
        {
            bf::path p(filename);
            if (bf::exists(p))
            {
                ifstream myFile(filename.c_str(), ios::in | ios::binary);
                int length;
                myFile.read((char*)&length, sizeof(int));

                for (int i=0; i < length; i++)
                {
                    int rows, cols;
                    myFile.read((char*)&rows, sizeof(int));
                    myFile.read((char*)&cols, sizeof(int));
                    T *data = new T[rows*cols];
                    myFile.read((char*)data, sizeof(T)*rows*cols);
#if _DEBUG
                    //cout << data[0] << endl;
#endif
                    cv::Mat head(rows, cols, cv::DataType<T>::type, data);
                    vMat->push_back(head);
                }               
            }
            else
            {
                cerr << filename << " doesn't exist..." << std::endl;
            }
        }

        template <class T>
        static __declspec(dllexport)
            void saveImgClusterToTrnFile(string filename, vector<cv::Mat> docs)
        {
            CV_Assert(docs.size() > 0);

            ofstream oFile(filename.c_str(), ios::out);
            oFile << docs.size() << "\n";

            for (int i=0; i< docs.size(); i++)
            {
                cv::Mat doc = docs[i];
                for (int j=0; j< doc.rows; j++)
                {
                    oFile << doc.at<T>(j, 0) << " ";
                }
                oFile << "\n";
            }
            oFile.close();	
        }
        static __declspec(dllexport)
            inline string zeros(int length, int idx)
        {
            char buffer[128];
            string zeros = "0";
            int n;
			if(idx > 0)
				n=log10(idx) + 1;
			else
			{
				n = 0;
			}
            for (int z = 0; z < length - n; z++)
            {
                zeros = zeros + "0";
            }
            return zeros;
        }

        static __declspec(dllexport) int DirectoryChecker(string dir);

        static __declspec(dllexport) string DirectoryBuilder(string dir);

        static __declspec(dllexport) vector<string> FileReader(string foldername, string type);

        static __declspec(dllexport) vector<string> FolderReader(string dir);

        static __declspec(dllexport) void GpsDataReader(string filename, cv::Mat *o_gpsdata);

        static __declspec(dllexport) void ThetaDataReader(string filename, cv::Mat *o_thetaMat);

        static __declspec(dllexport) string FileExtensionChange(string filename, string extension);

        static __declspec(dllexport) void imageIdxExtractor(
            std::string filename,
            int *idx,
            int *view_idx);

        static __declspec(dllexport) CubeMat CubeMatBuilder(string foldername);

        static __declspec(dllexport) CubeMat CubeMatBuilderFrFolder(string foldername);

        static __declspec(dllexport) CubeMat CubeMatSampling(string foldername, int steps);

        static __declspec(dllexport) void file_name_splitter(
            string filename, 
            vector<string> *o_name_list,
            string delimiter);

        static __declspec(dllexport) int FileIdxExtractor(string filename);

        static __declspec(dllexport) vector<TstImg> XmlReader(string xml_file);

        static __declspec(dllexport) CubeMat SearchResultGroup(vector<string> dist_list, vector<string> index_list);

        static __declspec(dllexport) bool copyFile(string src, string dst);        

        template <class T, class S>
        static __declspec(dllexport) 
            void SaveTestImgDescriptor(vector<testImgSift> *vtis, SearchParams params)
        {
            CV_Assert(vtis->size() > 0);
            FileManager fm;

            string testimgLog = params.testImgFolder + "\\test.xml";

            cv::FileStorage fs(testimgLog, cv::FileStorage::WRITE);
            fs << "images" << "{";
            // Save test image name and ground truth latitude and longitude
            for (int i = 0; i < vtis->size(); i++)
            {
                testImgSift tis = (*vtis)[i];
                char buffer[256];
                sprintf(buffer, "id_%d", i);

                string filename = params.testImgFolder + "\\" + (*vtis)[i].imgname + ".dsp";
                string tagFilename = params.testImgFolder + "\\" + (*vtis)[i].imgname + ".tag";
                string wdFilename = params.testImgFolder + "\\" + (*vtis)[i].imgname + ".wd";

                fs << static_cast<string>(buffer) << "{";
                fs << "name" << "[" << tis.imgname<< "]";
                fs << "descriptor" << "[" << filename << "]";
                fs << "tag" << "[" << tagFilename << "]";
                fs << "word" << "[" << wdFilename << "]";
                fs << "glat" << tis.glat;
                fs << "glng" << tis.glng;
                fs << "}";
                fm.SaveMat2Disk<T>((*vtis)[i].descriptor, filename);
                fm.SaveMat2Disk<S>((*vtis)[i].voting_tag, tagFilename);
                fm.SaveMat2Disk<T>((*vtis)[i].words, wdFilename);
            }

            fs << "}";
            fs.release();
        }

        template <class T, class S>
        static __declspec(dllexport)
            vector<testImgSift> ReadTestImgDescriptor(SearchParams params)
        {
            ImageManager im;
            FileManager fm;
            vector<testImgSift> testList;

            cv::FileStorage fs(params.testImgFolder + "\\test.xml", cv::FileStorage::READ);
            cv::FileNode images_node = fs["images"];
            cv::FileNodeIterator img_node_it = images_node.begin();
            cv::FileNodeIterator img_node_end = images_node.end();

            for (; img_node_it != img_node_end; img_node_it++)
            {
                cv::FileNode id_node = *img_node_it;
                //cout << (string)id_node["name"] << endl;
                testImgSift tis;
                tis.imgname = (string)id_node["name"];
                cv::Mat descriptor, tag, word;
                fm.ReadMatFromDisk<T>(id_node["descriptor"], &descriptor);
                tis.descriptor = descriptor;
                fm.ReadMatFromDisk<S>(id_node["tag"], &tag);
                tis.voting_tag = tag;
                fm.ReadMatFromDisk<T>(id_node["word"], &word);
                tis.words = word;

                tis.glat = id_node["glat"];
                tis.glng = id_node["glng"];

                testList.push_back(tis);
            }

            return testList;
        }

        static __declspec(dllexport)
            inline void WriteTimeLog(string msg, string filename)
        {
            std::ofstream oFile(filename, ios::app);
            if (oFile.is_open())
            {
                oFile << msg << "\n";
                oFile.close();
            }
            else
            {
                std::cerr << "Fail to open file: " << filename << std::endl;
            }
        }

        static __declspec(dllexport)
            int SaveMatlabMatfile2Disk(cv::Mat data_m, string filename);

        static __declspec(dllexport)
            int ReadMatlabMatfrDisk(string matFilename, cv::Mat *oMat);

		static __declspec(dllexport)
			void GenerateGTXmlFile(vector<TstImg> testImgList, string filename);

    private:
        struct stat status;
    };

}
