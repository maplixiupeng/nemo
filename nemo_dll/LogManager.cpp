#include "LogManager.h"

using namespace nemo;

void LogManager::LogRecorder(string s_message)
{
	wstring a= log4cplus::helpers::towstring(s_message.c_str());
	
	log4cplus::helpers::LogLog::getLogLog()->setInternalDebugging(true);
	log4cplus::SharedAppenderPtr append_1(
		new log4cplus::RollingFileAppender(a));
	append_1->setName(LOG4CPLUS_TEXT("First"));

	log4cplus::tstring pattern = LOG4CPLUS_TEXT("%d{%m/%d/%y %H:%M:%S,%Q} [%t] %-5p %c{2} %%%x%% - %m [%l]%n");

	append_1->setLayout(std::auto_ptr<log4cplus::Layout>(new log4cplus::PatternLayout(pattern)));
	log4cplus::Logger::getRoot().addAppender(append_1);

	log4cplus::Logger root = log4cplus::Logger::getRoot();
	log4cplus::Logger test = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("test"));

	log4cplus::Logger subTest = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("test.subtest"));

	for(int i=0; i<1000; ++i) {
		log4cplus::NDCContextCreator _context(LOG4CPLUS_TEXT("loop"));
		LOG4CPLUS_DEBUG(test, a);
		
	}

}

/*!
	\fn void LogManager::EventLogRecorder
Parameters:
	@param 1: logFilename
	@param 2: log_message
	@param 3: log_context
				who  is the log writer
	@param 4: log_type
			instance 
*/
void LogManager::EventLogRecorder(string logFilename, string log_message, string log_context, string log_type)
{
	wstring w_filename = log4cplus::helpers::towstring(logFilename.c_str());
	wstring w_log_message = log4cplus::helpers::towstring(log_message.c_str());
	wstring w_log_context = log4cplus::helpers::towstring(log_context.c_str());
	wstring w_log_type = log4cplus::helpers::towstring(log_type.c_str());

	log4cplus::helpers::LogLog::getLogLog()->setInternalDebugging(true);
	log4cplus::SharedAppenderPtr appender(
		new log4cplus::RollingFileAppender(w_filename));
	appender->setName((LOG4CPLUS_TEXT("Event")));

	log4cplus::tstring pattern = LOG4CPLUS_TEXT("%d{%m/%d/%y %H:%M:%S,%Q} [%t] %-5p %c{2} %%%x%% - %m [%l]%n");

	appender->setLayout(std::auto_ptr<log4cplus::Layout>(new log4cplus::PatternLayout(pattern)));
	log4cplus::Logger::getRoot().addAppender(appender);
	log4cplus::Logger root = log4cplus::Logger::getRoot();
	log4cplus::Logger type = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("type"));
	log4cplus::NDCContextCreator _context(w_log_context);
	LOG4CPLUS_DEBUG(type, w_log_message);
}

void LogManager::TimeLogRecorder(string logFilename, string log_message, string log_context, string log_type, double elapse_time)
{
	wstring w_filename = log4cplus::helpers::towstring(logFilename.c_str());
	wstring w_log_message = log4cplus::helpers::towstring(log_message.c_str());
	wstring w_log_context = log4cplus::helpers::towstring(log_context.c_str());
	wstring w_log_type = log4cplus::helpers::towstring(log_type.c_str());

	log4cplus::helpers::LogLog::getLogLog()->setInternalDebugging(true);
	log4cplus::SharedAppenderPtr appender(
		new log4cplus::RollingFileAppender(w_filename));
	appender->setName((LOG4CPLUS_TEXT("Time")));
	log4cplus::tstring pattern = LOG4CPLUS_TEXT("%d{%m/%d/%y %H:%M:%S,%Q} [%t] %-5p %c{2} %%%x%% - %m [%l]%n");

	appender->setLayout(std::auto_ptr<log4cplus::Layout>(new log4cplus::PatternLayout(pattern)));
	log4cplus::Logger::getRoot().addAppender(appender);
	log4cplus::Logger root = log4cplus::Logger::getRoot();
	log4cplus::Logger type = log4cplus::Logger::getInstance(LOG4CPLUS_TEXT("type"));
	log4cplus::NDCContextCreator _context(w_log_context);
	LOG4CPLUS_DEBUG(type, w_log_message);
}

void LogManager::SiftCollectorReadMe(FeatureParams param)
{
	wstring w_filename = LOG4CPLUS_TEXT("README.txt");

	log4cplus::helpers::LogLog::getLogLog()->setInternalDebugging(true);
	log4cplus::SharedAppenderPtr appender(new log4cplus::RollingFileAppender(w_filename));
	appender->setName((LOG4CPLUS_TEXT("Readme")));
}