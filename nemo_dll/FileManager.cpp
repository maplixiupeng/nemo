#include "FileManager.h"

using namespace nemo;

int FileManager::DirectoryChecker(string dir)
{
    bf::path p(dir);

    if(bf::exists(p))
    {
        if(bf::is_regular_file(p))
            return _IS_FILE;
        if (bf::is_directory(p))
        {
            return _IS_FOLDER;
        }
    }
    else
        return _NOT_EXISTED;
    /*if (access(dir.c_str(), 0) == 0)
    {
    stat(dir.c_str(), &status);
    if (status.st_mode * S_IFDIR)
    {
    return _IS_FOLDER;
    }
    else
    return _IS_FILE;
    }
    else
    return _NOT_EXISTED;	*/
}

/*!
\fn nemo::lib_FileProcessor::directoryBuilder(string dir)
Parameters:
@param 1: string dir
build directory under current work directory, dir should end with "\\" e.g. tina\\test\\
return:
complete directory created

Build directory under current working directory
*/
string FileManager::DirectoryBuilder(string dir)
{
    char buffer[256];
    vector<string> _sv;
    string subfile;

    size_t found = dir.find("\\");
    size_t found_temp = 0;
    size_t start = 0;
    int i = 1;

    while(found <= dir.length())
    {		
        int size = found - start;
        string s = dir.substr(start, size);
        _sv.push_back(s);
        found_temp = found;
        start = found_temp + 1;
        found = dir.find("\\", start);		
        i++;
    }

    bf::path p(dir);
    string foldername;
    if (p.is_absolute())
    {
        foldername = "";
    }
    else
    {
        getcwd(buffer, 256);
        foldername = static_cast<string>(buffer);
        foldername.append("\\");
    }

    for(int i = 0; i < _sv.size(); i++)
    {		
        foldername.append(_sv[i]);
        switch(int i = DirectoryChecker(foldername))
        {
        case _IS_FOLDER:
            break;
        case _IS_FILE:
            mkdir(foldername.c_str());
            break;
        case _NOT_EXISTED:
            mkdir(foldername.c_str());
            break;
        }
        foldername.append("\\");
    }
    return foldername;
}

/*!
\fn nemo::lib_FileProcessor::fileReader(string foldername, string type)

Parameters: 	
@param 1: string foldername, ex. ..\\videos\\GoogleStreetView\\0
@param 2: string type, "\\*.jpg"

Return
vector<string> 

Read files list under "foldername", return file list under the given folder
*/
vector<string> FileManager::FileReader(string foldername, string type)
{
    _finddata_t file;		
    long lf;
    vector<string> _v;
    string currdir = foldername;

    if(DirectoryChecker(foldername) != _IS_FOLDER)
    {
        cout << foldername <<" is not a exited folder" << endl;
        return _v;
    }


    const char *filename = currdir.append(type).c_str();
    if((lf = _findfirst(filename, &file)) == -1)
        cout << "There is no files under the director of " << foldername << endl;
    else
    {		
        string _filename = foldername;
        _v.push_back(_filename.append("\\").append(file.name));
        while(_findnext(lf, &file) == 0)
        {
            string _filename = foldername;
            _v.push_back(_filename.append("\\").append(file.name));
        }
    }
    return _v;
}

/*!
\fn  nemo::lib_FileProcessor::folderReader(string dir)
Parameters:
@param 1: string dir

return:
vector<string> folder name list contained in given directory

Retrieve all folder under dir
*/
vector<string> FileManager::FolderReader(string dir)
{
    vector<string> _vfolder;
    struct _finddata_t filefind;
    string curr = dir + "\\*.*";
    int done = 0, i, handle;
    if((handle = _findfirst(curr.c_str(), &filefind)) == -1) 
    {
        return _vfolder;
        //		return;
    }

    while(!(done = _findnext(handle, &filefind)))
    {
        if(!strcmp(filefind.name, ".."))
            continue;		
        if((_A_SUBDIR == filefind.attrib))
        {
            //xcout << filefind.name << "(dir)" << endl;
            //xcurr = dir + "\\" + filefind.name;			
            _vfolder.push_back(dir + "\\" + filefind.name);
        }
        else
        {
            //cout << filefind.name << endl;
        }
    }

    _findclose(handle);
    return _vfolder;
}

/*!
\fn nemo::lib_FileProcessor::gps_data_reader(string filename, cv::Mat o_gps_data)
Parameters:
@param 1: string filename
gps data file name, ex. "post.txt"
@param 2: cv::Mat o_gps_data 
output mat contain gps data, column meaning "Id Lat Lng Alt q0 q1 q2 q3"
return:
none

Read gps data into Mat
*/
void FileManager::GpsDataReader(string filename, cv::Mat *o_gpsdata)
{
    int filestate = DirectoryChecker(filename);
    if (filestate == _IS_FILE)
    {

        FILE *file;
        file = fopen(filename.c_str(), "r");

        //typedef vector<boost::iterator_range<string::iterator>> find_vector_type;
        //find_vector_type FindVec;
        typedef vector<string> split_vector_type;
        split_vector_type SplitVec;

        char line[1024];
        bool isFirstLine = true;
        int row_length = 0;
        int col_length = 0;

        vector<vector<double>> _data;	

        while(fgets(line, sizeof(line), file) != NULL)
        {
            string sline = static_cast<string>(line);
            boost::split(SplitVec, sline, boost::is_any_of(" "), boost::algorithm::token_compress_on);
            vector<double> _row; //Record each line data

            /*
            First line is title line, omit it
            */
            if(isFirstLine)
            {
                isFirstLine = false;
                col_length = SplitVec.size();
            }
            else
            {
                for(int i = 0; i < SplitVec.size(); i++)
                {
                    double f = atof(SplitVec[i].c_str());
                    _row.push_back(f);
                }
                _data.push_back(_row);
            }
            row_length++;
        }
        fclose(file);

        /*
        Read data into Mat
        */
        //(*o_gps_data).create(cv::Size(col_length, row_length), CV_32FC1);
        (*o_gpsdata).create(cv::Size(col_length, row_length-1), CV_64FC1);
        for(int row = 0; row < _data.size(); row++)
        {
            vector<double> _row = _data[row];
            for(int col = 0; col < _row.size(); col++)
            {
                //(*o_gps_data).at<float>(row, col) = _row[col];
                (*o_gpsdata).at<double>(row, col) = _row[col];
            }
        }	
    }
    else
    {
        cerr << filename << " doesn't exist!" << endl;
    }
}

void FileManager::ThetaDataReader(string filename, cv::Mat *o_thetaMat)
{
    int filestate = DirectoryChecker(filename);
    if (filestate == _IS_FILE)
    {
        FILE *file;
        file = fopen(filename.c_str(), "r");

        typedef vector<string> split_vector_type;
        split_vector_type SplitVec;

        char line[4096];
        bool isFirstLine = true;
        int row_length = 0;
        int col_length = 0;

        vector<vector<double>> _data;

        while(fgets(line, sizeof(line), file) != NULL)
        {
            string sline = static_cast<string>(line);
            boost::split(SplitVec, sline, boost::is_any_of(" "), boost::algorithm::token_compress_on);
            vector<double> _row; //Record each line data
            for(int i = 0; i < SplitVec.size(); i++)
            {
                double f = atof(SplitVec[i].c_str());
                _row.push_back(f);
            }
            _data.push_back(_row);
            row_length++;
            col_length = SplitVec.size() -1;
        }
        fclose(file);

        /*
        Read data into Mat
        */
        (*o_thetaMat).create(cv::Size(col_length, row_length), CV_32FC1);
        for(int row = 0; row < _data.size(); row++)
        {
            vector<double> _row = _data[row];
            for(int col = 0; col < _row.size()-1; col++)
            {
                //(*o_gps_data).at<float>(row, col) = _row[col];
                (*o_thetaMat).at<float>(row, col) = (float)_row[col];
            }
        }
    }
    else
    {
        cerr << filename << " doesn't exist!" << endl;
    }
}

/*!
\fn FileManager::FileExtensionChange(string filename, string extension)
Parameters:
@param 1: string filename
@param 2: string extension
return:
string
substitude extension of filename to a new extension
*/
string FileManager::FileExtensionChange(string filename, string extension)
{
    string _filename;
    bf::path p;
    p /= filename;
    string parent_path = p.parent_path().string();
    string file_stem = p.stem().string();
    _filename = parent_path + "\\" + file_stem + "." + extension;
    return _filename;
}

/*!
\fn UtilFileProcessor::imageIdxExtractor
Parameters:
@param 1: std::string filename
filename used to extract image index and view index, e.g. reprojected_img#_view#.jpg
@param 2: int *img_idx
output pointer to image index
@param 3: int *view_idx
output pointer to view index
return:
none

Use image filename to extract image index and view index
*/
void FileManager::imageIdxExtractor(std::string filename, int *img_idx, int *view_idx)
{
    boost::smatch what;

    boost::regex filename_reg(NEMO_IMAGE_NAME_REG);

    if (boost::regex_search(filename, what, filename_reg))
    {
        std::string str_img_idx = what[1];
        std::string str_view_idx = what[2];
        *img_idx = atoi(str_img_idx.c_str());
        *view_idx = atoi(str_view_idx.c_str()); 
    }
}

CubeMat FileManager::CubeMatBuilder(string foldername)
{
    CubeMat cm;
    vector<string> filelist = FileReader(foldername, "\\*.jpg");

    CV_Assert(filelist.size() > 0);
    vector<string> img_buffer;
    for (int i=0; i< filelist.size(); i++)
    {
        img_buffer.push_back(filelist[i]);
        if (i%4==3)
        {
            cm.push_back(img_buffer);
            img_buffer.clear();
        }
    }
    return cm;
}

CubeMat FileManager::CubeMatBuilderFrFolder(string foldername)
{
    CubeMat cm;
    vector<string> folderlist = FolderReader(foldername);
    CV_Assert(folderlist.size() > 0);

    for (int i=0; i < folderlist.size(); i++)
    {
        CubeMat _cm = CubeMatBuilder(folderlist[i]);

        for (int j=0; j< _cm.size(); j++)
        {
            cm.push_back(_cm[j]);
        }		
    }

    return cm;
}

CubeMat FileManager::CubeMatSampling(std::string foldername, int steps)
{
    //CubeMat cm = this->CubeMatBuilder(foldername);
    CubeMat cm = CubeMatBuilderFrFolder(foldername);

    CubeMat cm_sample;
    int step = steps;

    for (int i = 0; i < cm.size(); i++)
    {
        if (step == steps)
        {
            cm_sample.push_back(cm[i]);
            step = 0;
        }
        step++;
    }

    return cm_sample;
}

/*!
\fn FileManager::file_name_splitter
Parameters:
@param 1: string filename
@param 2: vector<string> *o_name_list
@param 3: string delimiter

return:
none

split string into word list accroding to given delimieter
*/
void FileManager::file_name_splitter(
    string filename, 
    vector<string> *o_name_list,
    string delimiter)
{
    boost::split(
        *o_name_list, 
        filename,
        boost::is_any_of(delimiter),
        boost::algorithm::token_compress_on);
}

/*!
\fn FileManager::FileIdxExtractor(string filename)
Parameters:
@param 1: string filename
return:
int index
*/
int FileManager::FileIdxExtractor(string filename)
{
    vector<string> _vfilename;
    file_name_splitter(filename, &_vfilename, "_");
    int idx = atoi(_vfilename[1].c_str());
    return idx;
}

vector<TstImg> FileManager::XmlReader(string xml_file)
{
    cv::FileStorage fs(xml_file, cv::FileStorage::READ);
    cv::FileNode node = fs["images"];

    cv::FileNodeIterator it = node.begin();
    cv::FileNodeIterator it_end = node.end();

    vector<TstImg> img_list;

    for (; it != it_end; it++)
    {
        TstImg ti;
        cv::FileNode fn = *it;
        cv::FileNodeIterator _it = fn.begin();
        cv::FileNodeIterator _it_end = fn.end();
        std::cout << (int)(*_it) << std::endl;
        ++_it;
        //std::cout << (string)(*_it) << std::endl;
        ti.filename = (string)(*_it);
        ++_it;
        ti.lat = (double)(*_it);
        ++_it;
        ti.lng = (double)(*_it);
        img_list.push_back(ti);
    }

    return img_list;
}

CubeMat FileManager::SearchResultGroup(vector<string> dist_list, vector<string> index_list)
{
    CV_Assert((dist_list.size() == index_list.size()) && dist_list.size() > 0 && index_list.size() > 0);

    CubeMat cm;

    for (int i=0; i<dist_list.size(); i++)
    {
        string dist_file = dist_list[i];
        vector<string> filename_list;
        file_name_splitter(dist_list[i], &filename_list, "_");
    }

    return cm;
}

bool FileManager::copyFile(string src, string dst)
{
    std::ifstream ifsrc;
    std::ofstream ofdst;

    ifsrc.open(src.c_str(), std::ios::binary);
    ofdst.open(dst.c_str(), std::ios::binary);
    if (!ifsrc.is_open() || !ofdst.is_open())
    {
        return false;
    }

    ofdst << ifsrc.rdbuf();
    ofdst.close();
    ifsrc.close();
    return true;
}

int FileManager::SaveMatlabMatfile2Disk(cv::Mat data_m, string ofilename)
{
    MATFile *pmat;
    mxArray *pa2;

    if (ofilename=="")
    {
        ofilename = "theta.mat";
    }

    data_m = data_m.t();
    int rows =data_m.rows;
    int cols = data_m.cols;
    pa2 = mxCreateDoubleMatrix(cols, rows, mxComplexity::mxREAL);
    if (pa2 == NULL)
    {
        printf("%s : Out of memory on line %d\n", __FILE__, __LINE__);
        printf("Unable to create mxArray.\n");
        return(EXIT_FAILURE);
    }

    const int _size_pdata = rows * cols;

    double *pdata;
    pdata = (double *)mxGetPr(pa2);

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

    int status;
    printf("Creating file %s...\n\n", ofilename.c_str());
    pmat = matOpen(ofilename.c_str(), "w");
    if (pmat == NULL) {
        printf("Error creating file %s\n", ofilename.c_str());
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
        printf("Error closing file %s\n",ofilename.c_str());
        return(EXIT_FAILURE);
    }

    printf("Done\n");
    //delete pdata;
    return(EXIT_SUCCESS);
}

int FileManager::ReadMatlabMatfrDisk(string sfile, cv::Mat *oMat)
{
    MATFile *pmat;
    const char **dir;
    const char *name;
    int ndir;
    int i;
    mxArray *pa;

    const char *file = sfile.c_str();

    printf("Reading file %s...\n\n", file);

    /*
    * Open file to get directory
    */
    pmat = matOpen(file, "r");
    if (pmat == NULL) {
        printf("Error opening file %s\n", file);
        return(1);
    }

    /*
    * get directory of MAT-file
    */
    dir = (const char **)matGetDir(pmat, &ndir);
    if (dir == NULL) {
        printf("Error reading directory of file %s\n", file);
        return(1);
    } else {
        printf("Directory of %s:\n", file);
        for (i=0; i < ndir; i++)
            printf("%s\n",dir[i]);
    }
    mxFree(dir);

    /* In order to use matGetNextXXX correctly, reopen file to read in headers. */
    if (matClose(pmat) != 0) {
        printf("Error closing file %s\n",file);
        return(1);
    }
    pmat = matOpen(file, "r");
    if (pmat == NULL) {
        printf("Error reopening file %s\n", file);
        return(1);
    }

    /* Get headers of all variables */
    printf("\nExamining the header for each variable:\n");
    for (i=0; i < ndir; i++) {
        pa = matGetNextVariableInfo(pmat, &name);
        if (pa == NULL) {
            printf("Error reading in file %s\n", file);
            return(1);
        }
        /* Diagnose header pa */
        printf("According to its header, array %s has %d dimensions\n",
            name, mxGetNumberOfDimensions(pa));
        if (mxIsFromGlobalWS(pa))
            printf("  and was a global variable when saved\n");
        else
            printf("  and was a local variable when saved\n");
        mxDestroyArray(pa);
    }

    /* Reopen file to read in actual arrays. */
    if (matClose(pmat) != 0) {
        printf("Error closing file %s\n",file);
        return(1);
    }
    pmat = matOpen(file, "r");
    if (pmat == NULL) {
        printf("Error reopening file %s\n", file);
        return(1);
    }

    /* Read in each array. */
    printf("\nReading in the actual array contents:\n");
    for (i=0; i<ndir; i++) {
        pa = matGetNextVariable(pmat, &name);
        if (pa == NULL) {
            printf("Error reading in file %s\n", file);
            return(1);
        } 
        /*
        * Diagnose array pa
        */
        printf("According to its contents, array %s has %d dimensions\n",
            name, mxGetNumberOfDimensions(pa));
        
        double *pdata;
        pdata = (double *)mxGetPr(pa);
        int rows = mxGetM(pa);
        int cols = mxGetN(pa);

        cv::Mat mData(cols, rows, CV_64FC1, pdata);
        *oMat = mData.t();

#if 0
        std::cout << oMat->col(1);
#endif


        if (mxIsFromGlobalWS(pa))
            printf("  and was a global variable when saved\n");
        else
            printf("  and was a local variable when saved\n");
        //mxDestroyArray(pa);
    }

    /*if (matClose(pmat) != 0) {
        printf("Error closing file %s\n",file);
        return(1);
    }*/
    printf("Done\n");
    return(0);
}

void FileManager::GenerateGTXmlFile(vector<TstImg> testImgList, string filename)
{
	CV_Assert(testImgList.size() > 0);
	cv::FileStorage fs(filename, cv::FileStorage::WRITE);
	fs << "images" << "{";
	for (int i=0; i < testImgList.size(); i++)
	{
		string imgName = testImgList[i].filename;
		double lat = testImgList[i].lat;
		double lng = testImgList[i].lng;
		
		char buffer[256];
		sprintf(buffer, "id_%d", i);
		fs << buffer << "{";
		fs << "name" << "\\" + imgName;
		fs << "lat" << lat;
		fs << "lng" << lng;
		fs << "}";
		
	}
	fs << "}";
	fs.release();
	std::cout << "Save all data to xml file ... ";
}