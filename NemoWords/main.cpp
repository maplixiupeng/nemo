#include "nemo_dll\common.h"
#include "WordUtilities.h"

using namespace nemo;

int main(int argc, char **argv)
{
	float fileSize;
 	string config_file;
	string featureFolder;
	double clusterAmt;

	string ref_imgFolder;
	string wdFilename;
	string wdFlannfilename;
	string invertFilename;
	string gps_filename;

	string testImgFolder;
	int type;

	bpo::options_description generic("Generic options");
	generic.add_options()
		("version,v", "print version string")
		("help", "Produce help message")
		("config,c", bpo::value<string>(&config_file)->default_value("config.cfg"), 
		"name of a file of a configuration.");
	
	bpo::options_description config("Configuration");
	config.add_options()
		("featureFolder", bpo::value<string>(&featureFolder)->default_value(""),
			"Feature folder")
		("clusterAmt", bpo::value<double>(&clusterAmt)->default_value(0), "Word amount")
		("ref_imgFolder", bpo::value<string>(&ref_imgFolder)->default_value(""), "Refere image folder")
		("gps_filename", bpo::value<string>(&gps_filename)->default_value(""), "Refere image folder")
		("wdFilename", bpo::value<string>(&wdFilename)->default_value(""), "word filename")
		("wdFlannfilename", bpo::value<string>(&wdFlannfilename)->default_value(""), "word flann filename")
		("invertFilename", bpo::value<string>(&invertFilename)->default_value(""), "word invert filename")
		("testImgFolder", bpo::value<string>(&testImgFolder)->default_value(""), "test image folder")
		("type", bpo::value<int>(&type)->default_value(0), "test type");	

	/*bpo::options_description hidden("Hidden options");
	hidden.add_options()
		("input-file", bpo::value<vector<string>>(), "input file");*/

	bpo::options_description cmdline_options;
	cmdline_options.add(generic).add(config);

	bpo::options_description config_file_options;
	config_file_options.add(config);
	
	bpo::variables_map vm;
	/*bpo::positional_options_description p;
	p.add("input-file", -1);	
	bpo::store(bpo::command_line_parser(ac, av).options(cmdline_options).positional(p).run(), vm);*/
	bpo::store(bpo::parse_command_line(argc, argv, generic), vm);
	bpo::notify(vm);  

	ifstream ifs(config_file.c_str());
	if (!ifs)
	{
		cout << "can not open config files: " << config_file << "\n";
		return 0;
	}
	else
	{
		bpo::store(parse_config_file(ifs, config_file_options), vm);
		notify(vm);
	}

	WordUtilities wu;
	//wu.wordGeneration(featureFolder, clusterAmt);
	//wu.word_entropy_builder(ref_imgFolder, wdFilename, wdFlannfilename);
	string gpsFilename = "C:\\TDDownload\\GoogleStreetview\\1210724640000\\DVD V1.7 Pittsburgh\\1210724640000\\pose.txt";
	//wu.invert_file_analysis("..\\x64\\Release\\words\\000000.wdFreq", gpsFilename);

	//vector<vector<long int> > v_invertfile = wu.read_word_invert("..\\x64\\Release\\words\\000000.wdFreq");

	//wu.generateModelTestImages(ref_imgFolder, gps_filename, 100, testImgFolder);
	wu.search_by_word(wdFilename, testImgFolder, invertFilename, gps_filename, type, ref_imgFolder);
	//FileManager fm;
	//cv::Mat sift;
	//fm.ReadMatFromDisk<float>("..\\x64\\Release\\amirSC_3\\00001.sift", &sift);
	
}