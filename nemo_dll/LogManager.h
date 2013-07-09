#pragma once
#include "common.h"

namespace nemo
{

	class LogManager
	{
	public:
		LogManager(void){};
		~LogManager(void){};

		void LogRecorder(string logFilename);

		void EventLogRecorder(string logFilename, string log_message, string log_context, string log_type);

		void TimeLogRecorder(string logFilename, string log_message, string log_context, string log_type, double elapse_time);

		void SiftCollectorReadMe(FeatureParams param);
	};

}
