//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file Logger.h
/// @brief An helper class to log, very easy to use. 
///
/// Redirect the log content to a FILE handle or a file. 
///
/// @version 1.0
/// @author Junqiao Wang
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////

#ifndef	LOGGER_H
#define LOGGER_H

#include <assert.h>
#include <stdarg.h>
#include <time.h>

#include <string>

#include "Common.h"


/// @brief A logger class used to define the log operation.
///
/// Redirect the log content to a FILE handle or a file. 
class Logger {
public:
	/// @brief Constructor of a logger.
	///
	/// Construct an instance of Logger use a FILE pointer.
	///
	/// @param[in] fp: the pointer of a output stream buffer.
	/// @return NULL
	explicit Logger(FILE *fp, Logger *logger = NULL) {
		this->fp = fp;
		this->logger = logger;
		//assert(fp != NULL);
	}

	/// @brief Constructor of a logger.
	///
	/// Construct an instance of Logger use a log file name and a bool value append, which 
	/// means whether the content will append in the end of the original file or not.
	///
	/// @param[in] file_name: The log file name we want to write the log into;
	/// @param[in] append: Whether the log content appends in the end of the original file or not.
	/// @return NULL
	/// @note 1. The file_name must be a legal file path.

	
	Logger(const char* file_name, bool append, Logger *logger = NULL) {
		if(append) {
			this->fp = fopen(file_name, "a");
		} else {
			this->fp = fopen(file_name, "w");
		}
		this->logger = logger;
		//assert(this->fp != NULL);
	}

	/// @brief Print the log content like standard C IO function printf.
	///
	/// This function is very easy to use, if you're familiar with standard C IO function "printf".
	///
	/// @param[in] format: The output format.
	/// @param[in] ...: The variable arguments list
	/// @return NULL
	void printf(const char* format, ...) {
		static char buf[5000], str[5000];
		va_list ap;
		va_start(ap, format);
		vsprintf(buf, format, ap);
		va_end(ap);
		sprintf(str, "[%s] - %s\n", get_time().c_str(), buf);
		this->output(str);
	}
	~Logger() {
		if(this->fp != NULL) {
			fflush(this->fp);
		}
	}

private:
	FILE *fp;
	Logger *logger;

	void output(const char* str) {
		fprintf(this->fp, "%s", str);
		if (logger) {
			logger->output(str);
		}
	}

	static std::string get_time() {
		char ch[100];
		time_t t_t = time(NULL);
		tm *tt = gmtime(&t_t);
		sprintf(ch, "%d-%02d-%02d %02d:%02d:%02d", 
			tt->tm_year+1900, tt->tm_mon+1, tt->tm_mday, 
			tt->tm_hour+8, tt->tm_min, tt->tm_sec);
		return ch;
	}
	DISALLOW_COPY_AND_ASSIGN(Logger);
};



#endif