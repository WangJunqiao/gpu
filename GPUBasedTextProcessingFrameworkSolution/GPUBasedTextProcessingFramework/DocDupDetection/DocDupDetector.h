//////////////////////////////////////////////////////////////////////////
//////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file DocDupDetector.h
/// @brief A virtual class declare some primary functions of a document duplicate detector
///
/// Document Duplicate Detector is an algorithm which is used to detect highly similar document pairs 
/// in a big document set 
///
/// @version 1.0
/// @author Junqiao Wang
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////
//////
#ifndef DOCDUPDETECTOR_H
#define DOCDUPDETECTOR_H

#include <vector>
#include <time.h>
using namespace std;


/// @brief A virtual class declare some primary functions of a document duplicate detector
///
/// Document Duplicate Detector is an algorithm which is used to detect highly similar document pairs 
/// in a big document set 
class DocDupDetector{
public:
	/// @brief some initialize operation before add document.
	/// @return NULL
	virtual void initialize() = 0;

	/// @brief Add a document into the document pool, which is a space separated English words. 
	/// e.g. "I have a dream, that one day I can run a company like gg."
	/// @return NULL
	virtual void add_document(string doc) = 0;
	
	/// @brief The core part of document duplicate detection. We use "fuzzy hashing" to reduce
	/// the calculation. For example we have these documents: 
	/// doc1 = "aaaaaaaaaa"
	/// doc2 = "aaaaaabaaa"
	///	doc3 = "aaaabbbbbb"
	///	doc4 = "aaaacbbbbc" 
	/// after we run the core process, we will get the potential duplicate document pairs {<doc1, doc2>, <doc3, doc4>}
	/// @return NULL
	virtual void calculate_dups() = 0;

	/// @brief Get the potential duplicate documents of doc_id.
	/// @param [in] doc_id: the document's id you want to check.
	/// @return The set of document ids.
	virtual vector<int> get_candidate_dup_docs(int doc_id) = 0;

	/// @brief Calculate the real duplicate documents of every document.
	/// @return NULL.
	virtual void refine() = 0;

	/// @brief get the real duplicate document set of document doc_id.
	/// @return The set of document ids.
	virtual vector<int> get_real_dup_docs(int doc_id) = 0;

	/// @brief Calculate the score of two document. The higher the degree of duplication, the bigger the value of score.
	/// @return An integer score value between [0, 100]
	virtual int score(const char *doc_a, const char *doc_b);

	/// @brief Calculate the edit distance of two document.
	/// @return An integer indicate the edit distance.
	int edit_dist(const char *str1, const char *str2);

	virtual ~DocDupDetector() {

	}
	
	clock_t core_time;
};

#endif