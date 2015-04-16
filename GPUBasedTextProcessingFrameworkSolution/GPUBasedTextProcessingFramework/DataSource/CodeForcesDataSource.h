//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file CodeForcesDataSource.h
/// @brief A class which derives from DocumentSource defines a text document stream. 
///
/// This class read text file from a folder. Document Duplicate Detection algorithm will use this document source.
///
/// @version 1.0
/// @author Junqiao Wang
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////

#ifndef CODEFORCESDATASOURCE_H
#define CODEFORCESDATASOURCE_H
#include "DocumentSource.h"

//#include <io.h>
//#include <direct.h>

#include <sys/types.h>
#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <iostream>
#include <string>
#include <map>

using namespace std;

map<int,string> document;

/// @brief A class which derives from DocumentSource defines a text document stream. 
///
/// This class read text file from a folder. Document Duplicate Detection algorithm will use this document source.
class CodeforcesDataSource : public DocumentSource {
public:
	string dir;
	//_finddata_t fileinfo; 
	
    DIR* files;
    string next_file_name;

    map<int, string> document;
	map<int, vector<int> > afterRefineCandPairs;
	int flag;
	int number;
	intptr_t iFind;

	bool openSource();
	void set_files_directory(string dir) {
		this->dir = dir;
	}
	bool hasNext();
	//	CodeforcesDataSource();
	string getNextDocument();
	void closeSource() {

	}

	void setDocument(int id,string hash_value);
	string getDocumentName(int id);
	~CodeforcesDataSource();
};


//implement of class CodeforcesDataSource

bool CodeforcesDataSource::openSource() {
	// string dir="";
	files = opendir(dir.c_str());
    /*if (_access(dir.c_str(), 06) == -1)  
	{  
		cerr << "error: directory does not exist." << endl;  
		exit(-1);  
	} */ 

	if (dir.at(dir.length() - 1) != '\\')  
	{  
		dir += '\\';  
	}  
/*
	if (_chdir(dir.c_str()) != 0)  
	{  
		cerr << "error: function _chdir() failed.";  
		exit(-1);  
	}  
	memset(&fileinfo, 0x0, sizeof(fileinfo));
	flag=0;
	number=0;*/
	return true;
}

bool CodeforcesDataSource::hasNext() {
	dirent *tmp;
    while ((tmp = readdir(files)) != NULL) {
        next_file_name = tmp->d_name;
        if (next_file_name.length() < 4) {
            continue;
        }
        if (next_file_name.substr(next_file_name.length() - 4) == ".txt") {
            return true;
        }
    }
    return false;
    /*
    if(flag==0){
		iFind = _findfirst("*.txt", &fileinfo);
		flag=1;
		if(iFind!=-1)
			return true;
	}
	else{
		if(_findnext(iFind, &fileinfo)==0)
			return true;
	}
	return false; */
}

string CodeforcesDataSource::getNextDocument() {
    string filePath(dir + next_file_name);
	
	printf("file = %s\n", filePath.c_str());

	static char content[1000000];
	FILE *fp = NULL;
	if((fp = fopen(filePath.c_str(), "rb")) == NULL){
		printf("could not open file - %s!\n", filePath.c_str());
	}
	long lsize;
	fseek(fp, 0, SEEK_END);
	lsize = ftell(fp);

	fseek(fp, 0, SEEK_SET);
	fread(content, sizeof(char), lsize, fp);
	content[lsize] = '\0';
	fclose(fp);
	
	document[number ++] = next_file_name;
	return content;
}
CodeforcesDataSource::~CodeforcesDataSource() {

}
void CodeforcesDataSource::setDocument(int id,string hash_value){
	document[id] = hash_value;
}
string CodeforcesDataSource::getDocumentName(int id){
	return document[id];
}
#endif
