#include "nemo_dll/common.h"
#include "GroupModel.h"

using namespace nemo;

void paramaterPrint(GroupParams gp)
{
	std::cout << "Image folder: " << std::endl;
	std::cout << "\t" << gp.imgFolder << std::endl;
	std::cout << "Output folder: " << std::endl;
	std::cout << "\t" << gp.outputFolder << std::endl;
	std::cout << "Theta file name: " << std::endl;
	std::cout << "\t" << gp.thetaFilename << std::endl;
	std::cout << "Step: " << std::endl;
	std::cout << "\t" << gp.step << std::endl;
	std::cout << "Group: " << std::endl;
	std::cout << "\t" << gp.group << std::endl;
	std::cout << "Fuzzy ratio: " << std::endl;
	std::cout << "\t" << gp.fuzzyRatio << std::endl;
}

int main(int argc, char **argv)
{
	string config_file;

	string imgFolder;
	string thetaFilename;
	string fuzzyMatlabClusterfilename;
	string fuzzyMatlabClusterCenterfilename;
	string outputFolder;

	int step;
	int group;
	int type;
	double fuzzyRatio;


	bpo::options_description generic("Generic options");
	generic.add_options()
		("version,v", "print version string")
		("help", "Produce help message")
		("config,c", bpo::value<string>(&config_file)->default_value("config.cfg"), 
		"name of a file of a configuration.");

	bpo::options_description config("Configuration");
	config.add_options()
		("imgFolder", bpo::value<string>(&imgFolder)->default_value(""), "Image folder")
		("thetaFilename", bpo::value<string>(&thetaFilename)->default_value(""), "feature folder")
		("fuzzyMatlabClusterfilename", bpo::value<string>(&fuzzyMatlabClusterfilename)->default_value(""), "fuzzy matlab cluster filename")
		("fuzzyMatlabClusterCenterfilename", bpo::value<string>(&fuzzyMatlabClusterCenterfilename)->default_value(""), "fuzzy matlab cluster center filename")
		("outputFolder", bpo::value<string>(&outputFolder)->default_value(""), "Group output folder")
		("step", bpo::value<int>(&step)->default_value(1), "Sampling step")
		("group", bpo::value<int>(&group)->default_value(1), "How many groups?")
		("type", bpo::value<int>(&type)->default_value(0), "Grouping type, 0: hard cluster, 1: soft cluster")
		("fuzzyRatio", bpo::value<double>(&fuzzyRatio)->default_value(0), "Fuzzy Ratio");	

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

	GroupParams gp;
	gp.imgFolder = imgFolder;
	gp.outputFolder = outputFolder;
	gp.thetaFilename = thetaFilename;
	gp.step = step;
	gp.group = group;
	gp.fuzzyMatlabfilename = fuzzyMatlabClusterfilename;
	gp.fuzzyMatlabClusterCenterfilename = fuzzyMatlabClusterCenterfilename;
	gp.fuzzyRatio = fuzzyRatio;

	paramaterPrint(gp);
	GroupModel gm;
	switch (type)
	{
	case 0:
		gm.GroupingDBFrTheta(gp);
		break;
	case 1:
		gm.FuzzyGroupDBFrTheta(gp);
		break;
	default:
		break;
	}





}