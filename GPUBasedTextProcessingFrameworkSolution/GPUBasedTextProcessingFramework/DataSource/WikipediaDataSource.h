//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file WikipediaDataSource.h
/// @brief A class which derives from DocumentSource retrive text document from a big text file. 
///
/// This class read text file from a big text file.
///
/// @version 1.0
/// @author Junqiao Wang
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////

#ifndef	WIKIPEDIADATASOURCE_H
#define WIKIPEDIADATASOURCE_H

#include "../Common/Common.h"
#include "../DataSource/DocumentSource.h"

#define MAX_DOC_LENGTH 5000000

/// @brief A class which derives from DocumentSource retrive text document from a big text file. 
///
/// This class read text file from a big text file.
class WikipediaDataSource : public DocumentSource {
public:
	explicit WikipediaDataSource(const string &file_name) {
		this->file_name = file_name;
		buf = new char[MAX_DOC_LENGTH];
		fp = NULL;
	}

	bool openSource() {
		if(fp != NULL) fclose(fp);
		fp = fopen(this->file_name.c_str(), "r");
		cc = 0;
		return fp != NULL;
	}

	bool hasNext() {
		if(cc > max_docs) return false;
		return fgets(buf, MAX_DOC_LENGTH, fp) != NULL;
	}

	std::string getNextDocument() {
		cc++;
		return buf;
	}

	void closeSource() {
		if(fp != NULL) {
			fclose(fp);
			fp = NULL;
		}
	}

	void set_max_docs(int x) {
		max_docs = x;
	}

	~WikipediaDataSource() {
		closeSource();
		delete[] buf;
	}

private:
	string file_name;
	FILE *fp;
	char *buf;
	int cc, max_docs;

	DISALLOW_COPY_AND_ASSIGN(WikipediaDataSource);
};

#endif