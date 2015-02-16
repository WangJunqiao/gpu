//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file IDFManager.h
/// @brief An IDFManager class to calculate tf-idf value
///
/// This file contains the IDFManager class, it can be either used
/// to calculate the idf value or tf-idf value.
///
/// @version 1.0
/// @author Bowen Liu
/// @date 01.13.2014
///
//////////////////////////////////////////////////////////////////////////
#ifndef IDFCALCULATOR_H
#define IDFCALCULATOR_H

#include <iostream>
#include <map>
#include <unordered_map>
#include <string>
#include <set>
#include <algorithm>
#include <cmath>
#include <vector>
#include <assert.h>
#include "./stemmer/stem_api.h"
#include "stop_words_list.h"
#include "../DataSource/DocumentSource.h"
#include "Logger.h"
using namespace std;

/// @var double MIN_INF
///
/// @brief The minimum minus number used in this file.
///
/// @warning the MIN_INF can only be used in this file!
static const double MIN_INF = -((long long)1)<<40;

/// @var double MAX_WORD_NUM
///
/// @brief The maximum word number used in this file.
///
/// @warning the MAX_WORD_NUM can only be used in this file!
static const int MAX_WORD_NUM = 1000000000;

/// @brief enum type to define calculation type of tf-idf
///
/// This enum defines types of calculating tf-idf which are there is a idf file
/// and the idf file does not exist.
enum CALC_TYPE{
	HAS_IDF_FILE,    //< enum: the idf file has already exit
	HAS_NO_IDF_FILE  //< enum: the idf file does not exist and need to be calculated
};

/// @brief An idf or tf-idf calculator class
///
/// calculate idf values or tf-idf values from a certain document source
/// sample use: DocumentSource doc_src = new DocumentSource(directory_name);
///				string out_file_path = "...";
///             Logger idf_logger(stdout);
///				IDFCalculator idf_calcor = new IDFCalculator(&idf_logger);
///				idf_calcor.calc_idf(doc_src, out_file_path.c_str());
///				delete idf_calcor;
/// @note 1. the class must pass a Logger object to the construct function
/// @note 2. all the idf values will be saved as format: word_name idf_value, 
/// all the tf-idf values will be saved as format: doc_id word_id tf_idf_value
class IDFManager{
public:
	explicit IDFManager(Logger *logger) {
		this->logger = logger;
		word_idf.clear();
	}

	/// @brief calculate idf value from the document source
	///
	/// calculate idf value from the document source, the idf value will be
	/// saved to file @ref out_idf_file and will be stored in the member variable @ref word_idf
	///
	/// @param[in] doc_src: the document source we are to calculate the idf value;
	/// @param[out] out_idf_file: file name you want to save the idf value to
	/// @return NULL
	/// @note 1. the result will be saved to file @ref out_idf_file,
	///	@note 2. this function will save all the words
	void calc_idf(DocumentSource *doc_src, const char *out_idf_file);

	/// @brief calculate idf value from the document source
	///
	/// calculate idf value from the document source, the idf value will be
	/// saved to file @ref out_idf_file but will not be stored in the member variable @ref word_idf
	///
	/// @param[in] doc_src: the document source we are to calculate the idf value;
	/// @param[out] out_idf_file: file name you want to save the idf value to
	/// @return NULL
	/// @note 1. the result will be saved to file @ref out_idf_file,
	///	@note 2. this function will save all the words
	void calc_idf(DocumentSource *doc_src, const int n, const char *out_idf_file);

	/// @brief calculate tf-idf value from the document source
	///
	/// calculate tf-idf from the document source, this function has two model:
	/// one is that the idf file has already exit so we just have to calculate the tf values;
	/// the other is that we have to both calculate the tf and idf values.
	///
	/// @param[in] doc_src: the document source we are to calculate the tf-idf value;
	/// @param[in] vocabulary_file: if the calc_type is HAS_IDF_FILE, then this is the input word-idf file;
	/// @param[in] calc_type: calculate type, see the @enum CALC_TYPE.
	/// @param[out] out_tf_idf_file: the output tf-idf file;
	/// @param[out] vocabulary_file: if the calc_type is HAS_NO_IDF_FILE, then this is the output vocabulary file;
	/// @return NULL
	/// @note 1. the input word-idf file must format as: word_name idf_value;
	///	@note 2. the output vocabulary file is sorted;
	/// @note 3. the output tf-idf file is format as: doc_id word_id tf_idf_value
	void calc_tfidf(DocumentSource *doc_src, const char *vocabulary_file, 
		const char* out_tf_idf_file, CALC_TYPE calc_type, int n = MAX_WORD_NUM);

	/// @brief load idf values from a certain file
	///
	/// load idf values from @ref file_name, the data format must be as follow: word_name idf_value 
	///
	/// @param[in] file_name: file name you want to load the idf value from
	/// @return NULL
	void load_idf(const char* file_name);

	/// @brief get word index from member variable @ref word_idf
	///
	/// get word index from the member variable @ref word_idf, you must first calculate the idf value
	/// or load the idf values from file use the function @ref load_idf
	///
	/// @param[in] word: word name which you want to get the index
	/// @return index of the word
	int get_word_id(const char *word);

	/// @brief get word idf from from member variable @ref word_idf
	///
	/// get word index from the member variable @ref word_idf, you must first calculate the idf value
	/// or load the idf values from file use the function @ref load_idf
	///
	/// @param[in] word: word name which you want to get the idf
	/// @return idf value of the word
	double get_word_idf(const char *word);

	/// @brief get word idf from from member variable @ref word_idf
	///
	/// get word index from the member variable @ref word_idf, you must first calculate the idf value 
	/// or load the idf values from file use the function @ref load_idf
	///
	/// @param[in] word_id: word index which you want to get the idf value
	/// @return idf value of the word index
	double get_word_idf(const int& word_id);

	/// @brief get word number from from member variable @ref word_idf
	///
	/// get word number from the member variable @ref word_idf, you must first calculate the idf value 
	/// or load the idf values from file use the function @ref load_idf
	///
	/// @return size of member variable @ref word_idf
	int get_word_num();

private:
	/// @brief calculate tf-idf from the document source
	///
	/// calculate tf-idf from the document source which the idf file has already exist,
	/// so we just have to calculate the tf then save the tf-idf;
	///
	/// @param[in] doc_src: the document source we are to calculate the td-idf value;
	/// @param[in] word_idf_file: the file is formated as <word_name	idf_value>, there is a Tab in the middle;
	/// @param[out] out_tf_idf_file: the output tf-idf file;
	/// @return NULL
	/// @note 1. this function is a internal function
	///	@note 2. he output tf-idf file is format as: doc_id	word_id	tf_idf_value
	void _calc_tfidf_with_idf(DocumentSource *doc_src, const char *word_idf_file, const char* out_tf_idf_file, int n);

	/// @brief calculate tf-idf from the document source
	///
	/// calculate tf-idf from the document source which the idf file does not exist,
	/// so we have to calculate the both the tf and idf values;
	///
	/// @param[in] doc_src: the document source we are to calculate the td-idf value;
	/// @param[out] out_vocabulary_file: the vocabulary file name we are to save the vocabulary to;
	/// @param[out] out_tf_idf_file: the output tf-idf file;
	/// @return NULL
	/// @note 1. this function is a internal function
	/// @note 2. the vocabulary is sorted
	///	@note 3. he output tf-idf file is format as: doc_id	word_id	tf_idf_value
	void _calc_tfidf_without_idf(DocumentSource *doc_src, const char *out_vocabulary_file, const char* out_tf_idf_file, int n);

	/// @brief wipe off all non- alphabet characters of a word
	///
	/// wipe off all non- alphabet characters of a word and connect the rest
	///
	/// @param[in] c: the word to be cleaned;
	/// @param[out] c: after cleaning, the result will still be saved in c;
	/// @return NULL
	/// @note 1. this function is a internal function
	/// @note 2. the output is also the input, both are "c".
	void _clean_word(char *c); 

	/// the word and its idf value
	vector<pair<string, double> > word_idf;
	/// log member
	Logger *logger;
};


#endif