#include "nemo_dll/common.h"
#include "FeatureModel.h"

#include <iostream>
#include <fstream>
#include <iterator>

using namespace std;
using namespace nemo;


int main(int argc, char** argv)
{
	float fileSize;
	string imgFolder;
	string gpsFilename;
	string featureType;
	string outputFolder;
	double siftBlurCof;
	string config_file;
	bool isAppend;
	int sampling;

	bpo::options_description generic("Generic options");
	generic.add_options()
		("version,v", "print version string")
		("help", "Produce help message")
		("config,c", bpo::value<string>(&config_file)->default_value("config.cfg"), 
		"name of a file of a configuration.");
	
	bpo::options_description config("Configuration");
	config.add_options()
		("imgFolder", bpo::value<string>(&imgFolder)->default_value(""),
			"Image folder")
		("gpsFilename", bpo::value<string>(&gpsFilename)->default_value(""), "GPS file name")
		("featureType", bpo::value<string>(&featureType)->default_value("SIFT"), "Feature point type, such as SIFT, SURF")
		("outputFolder", bpo::value<string>(&outputFolder)->default_value(""), "output feature files folder")
		("filesize", bpo::value<float>(&fileSize)->default_value(1), "file size")
		("siftBlurCof", bpo::value<double>(&siftBlurCof)->default_value(0.9), "Blue coefficient used in SIFT detection algorithm")
		("isAppend", bpo::value<bool>(&isAppend)->default_value(true), "Append new data to last existed Mat file")
		("sampling", bpo::value<int>(&sampling)->default_value(5), "Image cube sampling step");	

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

	/*if (vm.count("imgFolder"))
	{
		cout << "Include paths are: "
			<< vm["imgFolder"].as<string>() << endl;
	}

	if (vm.count("input-file"))
	{
		cout << "Input files are: "
			<< imgFolder << endl;
	}*/

	FeatureModel fm;	
	FeatureParams params;
	params.featureType = featureType;
	params.imgFolder = imgFolder;
	params.gpsFilename = gpsFilename;
	params.outputFolder = outputFolder;
	params.fileSize = fileSize;
	params.siftBlurCof = siftBlurCof;
	params.isAppend = isAppend;
	params.sampling = sampling;

	cout << "File Size is " << params.fileSize << endl;
	if (params.sampling == 0)
	{
		fm.FeatureCollectorGp(params);
	}
	else
	{
		fm.FeatureCubeCollector(params);
	}
	
	return 1;
}