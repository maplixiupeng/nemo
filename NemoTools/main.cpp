#include "nemo_dll/common.h"
#include "NemoUtilities.h"

using namespace nemo;

int main(int argc, char** argv)
{
	string config_file;

	int type;
	/*For type 0: Generate Matlab filename*/
	string thetafilename;
	string ofilename;

        // For image translator, type 1/2
        string corpusFoldername;
        string wdFilename;
        string trnFilename;

        //For type 3, image to word list
        string featureFolder;
        double wdAmt;

	// For type 4, word folders and 
	string imgListFolder;
	string wdFolder;

	bpo::options_description generic("Generic options");
	generic.add_options()
		("version,v", "print version string")
		("help", "Produce help message")
		("config,c", bpo::value<string>(&config_file)->default_value("config.cfg"), 
		"name of a file of a configuration.");

	bpo::options_description config("Configuration");
	config.add_options()
		("type", bpo::value<int>(&type)->default_value(1), "0: Matlab File translator, 1: image to wd list, 2: wdlist to trn file")
		("thetafilename", bpo::value<string>(&thetafilename)->default_value(""), "theta filename")
		("ofilename", bpo::value<string>(&ofilename)->default_value(""), "output mat filename")
                ("wdFilename", bpo::value<string>(&wdFilename)->default_value(""), "output word filename")
                ("trnFilename", bpo::value<string>(&trnFilename)->default_value(""), "output trn filename")
                ("featureFolder", bpo::value<string>(&featureFolder)->default_value(""), "Feature Folder")
                ("wdAmt", bpo::value<double>(&wdAmt)->default_value(0), "Word Amount")
				("imgListFolder", bpo::value<string>(&imgListFolder)->default_value(""), "Image folder")
				("wdFolder", bpo::value<string>(&wdFolder)->default_value(""), "word file folder");

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

	NemoUtilities nu;

	switch (type)
	{
	case 0:
		std::cout << "Translator theta file to Matlab mat..." << std::endl;
		nu.MatlabMatGenerator(thetafilename, ofilename);
                break;
        case 1:
            std::cout << "Translate Image to word list..." << std::endl;
            nu.corpus2WDList(corpusFoldername, wdFilename);
            break;
        case 2:
            std::cout << "Translate word list to trn filename..." << std::endl;
            nu.corpus2TrnDatafile(corpusFoldername, trnFilename);
            break;
		case 4:
			std::cout << "Draw Color Sift... " << std::endl;
			nu.DrawVisualWord(imgListFolder, wdFolder);
	default:
		break;
	}
}