////////////////////////////////////////////////////////////////////
//////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file DataSource.h
/// @brief A @ref DataSource class to read the raw documents from disk
///
/// This file contains a @ref DataSource class, you can pass the file path or directory path
/// into the construct method, then use the @ref openSource method to get the absolute path
/// of all files in the directory or get the path of the file if you pass a single file into
/// the construct method; you can use the @ref getNextDocument to get all the contents of next
/// document, but before you use @ref getNextDocument you must use @ref hasNext method to see
/// if there exits a "next document"; at last, you should use the @ref closeSource to close the
/// the @DataSource. After you open the data source you can also use the @ref getDocName to get
/// a certain document name, note that the doc_id when calculate the tf-idf is according to the 
/// order here.
///
/// @version 1.0
/// @author Bowen Liu
/// @date 01.17.2014
///
////////////////////////////////////////////////////////////////////
//////
#ifndef _KNN_DATA_SOURCE_H_
#define _KNN_DATA_SOURCE_H_

#include "../DataSource/DocumentSource.h"
#include <iostream>
#include <string>
#include <vector>
using namespace std;

/// @brief A DataSource class to read documents from disk
///
/// The objects of this class not only read single file but also can browse the
/// directory recursively.
/// sample use: DocumentSource *doc_src = new DocumentSource(directory_name);
///				if (doc_src->openSource()) {
///					...
///					string s;
///					if (doc_src->hasNext()) {
///						s = doc_src->getNextDocument();
///						...
///					}
///				}
///				doc_src->closeSource();
///				delete doc_src;
/// @note 1. you must first use the @ref hasNext before you use @ref getNextDocument
class DataSource : public DocumentSource {
public:
	DataSource(const char *path) {
		_path = path;
		_file_index = 0;
	}
	~DataSource(){}

	/// @brief open the document source
	///
	/// get all the file paths if the path is a directory path or the exact path passed in if
	/// it is a single file path, the path should pass into the construct method
	///
	/// @return true if the path is right
	bool openSource();

	/// @brief to see if there still exits files to be read
	///
	/// this method is to see if there still exists files to be read, you should use this method
	/// before you use the method @ref getNextDocument
	///
	/// @return true if there still exists files to be read
	bool hasNext();

	/// @brief get next document
	///
	/// this method is to get all the content of next document, before you use this method you should
	/// use @ref hasNext to see if there still exists documents
	///
	/// @return the content of the document
	string getNextDocument();

	/// @brief close the document source
	///
	/// close the document source, this method will do the cleaning things but you should still use 
	/// "delete" to delete the object if you use the "new"  
	///
	/// @return the content of the document
	void closeSource();

	/// @brief get a certain document name according to the document id
	///
	/// get a certain document name according to the document id, the id is used when calculate the tf-idf
	/// value of words
	///
	/// @return the document name
	string getDocName(const int &docid);

	/// @brief get all document names
	///
	/// get all document names of the directory
	///
	/// @return the document names
	vector<string> getDocNames();
private:
	FILE *fp;
	int _file_index;
	//path passed into the construct method
	string _path;
	//all the file paths of the directory
	vector<string> _doc_paths;
	//this method will browse the directory recursively, the @ref openSource call this method to
	//browse the directory
	bool _browse_dir(string path);
};

#endif