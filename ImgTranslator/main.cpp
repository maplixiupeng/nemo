#include "nemo_dll/common.h"
#include "NemoTranslator.h"

using namespace nemo;

int main(int argc, char **argv)
{
	string config_file;

	string corpusFoldername;
	string wdFilename;
	string trnFilename;
	int type;

	bpo::options_description generic("Generic options");
	generic.add_options()
		("version,v", "print version string")
		("help", "Produce help message")
		("config,c", bpo::value<string>(&config_file)->default_value("config.cfg"), 
		"name of a file of a configuration.");

	bpo::options_description config("Configuration");
	config.add_options()
		("corpusFoldername", bpo::value<string>(&corpusFoldername)->default_value(""), "Image folder")
		("wdFilename", bpo::value<string>(&wdFilename)->default_value(""), "feature folder")
		("type", bpo::value<int>(&type)->default_value(1), "0: image translation, 1: image wordlist to trn format")
		("trnFilename", bpo::value<string>(&trnFilename)->default_value(""), "feature folder");	

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

	NemoTranslator nt;
	switch (type)
	{
	case 0:
		nt.corpusTranslator(corpusFoldername, wdFilename);
		break;
	case 1:
		nt.corpusToTrnDataFile(corpusFoldername, trnFilename);
	}
	
}