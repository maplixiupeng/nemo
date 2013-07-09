#include "nemo_dll/common.h"
#include "SearchModel.h"

using namespace nemo;

int main(int argc, char **argv)
{
	string config_file;

	string outputFolder;
	string testImgFolder;
	string wordFolder;
	string trGpFolder;
	string modelFolder;
	string modelName;
	string amirFolder;
	int type;

	bpo::options_description generic("Generic options");
	generic.add_options()
		("version,v", "print version string")
		("help", "Produce help message")
		("config,c", bpo::value<string>(&config_file)->default_value("config.cfg"), 
		"name of a file of a configuration.");

	bpo::options_description config("Configuration");
	config.add_options()
		("outputFolder", bpo::value<string>(&outputFolder)->default_value(""), "Image folder")
		("testImgFolder", bpo::value<string>(&testImgFolder)->default_value(""), "feature folder")
		("type", bpo::value<int>(&type)->default_value(1), "0: Amire search, 1: Zheng Search")
		("wordFolder", bpo::value<string>(&wordFolder)->default_value(""), "feature folder")
		("trGpFolder", bpo::value<string>(&trGpFolder)->default_value(""), "training image group folder")
		("modelFolder", bpo::value<string>(&modelFolder)->default_value(""), "Topic Model folder")
		("modelName", bpo::value<string>(&modelName)->default_value("model-final"), "Topic Model name")
		("amirFolder", bpo::value<string>(&amirFolder)->default_value(""), "Amir folder");	

	bpo::options_description cmdline_options;
	cmdline_options.add(generic).add(config);

	bpo::options_description config_file_options;
	config_file_options.add(config);

	bpo::variables_map vm;

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

	SearchParams params;
	params.outputFolder = outputFolder;
	params.testImgFolder = testImgFolder;
	params.wordFolder = wordFolder;
	params.trGpFolder = trGpFolder;
	params.modelFolder = modelFolder;
	params.modelName = modelName;
	params.amirFolder = amirFolder;
	SearchModel sm;

	switch (type)
	{
	case 0:
		std::cout << "Doing Amir's search..." << std::endl;
		sm.TestAmirPlainSearch(params);
		break;
	case 1:
		std::cout << "Doing Zheng's search..." << std::endl;
		sm.TestImgGrooup(params);
		break;
	case 2:
		std::cout << "Doing Zheng's search in multiple scene groups..." << std::endl;
		sm.TestZhengSceneSearch(params);
		break;
	default:
		break;
	}
}