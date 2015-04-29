//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file MatrixFileReader.h
/// @brief A reader which is used to read the binary word similarity matrix file.
///
/// @version 1.0
/// @author Junqiao Wang
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////
#ifndef MATRIX_FILE_READER_H
#define MATRIX_FILE_READER_H

#include <vector>

#include "../Common/Logger.h"

/// @brief A reader which is used to read the binary word similarity matrix file.
///
///
class MatrixFileReader {
public:
	/// @brief Constructor of a MatrixFileReader.
	/// @param[in] logger: The Logger used to print log.
	/// @return NULL
	explicit MatrixFileReader(Logger *logger);

	/// @brief Initialize a MatrixFileReader use parameter matrix_file and word_file.
	/// @param[in] matrix_file: The file name of the matrix file.
	/// @param[in] word_file: The file name of the word file.
	/// @return A vector<int>: The i-th element of the integer vector means the number of words the i-th words is similar to. 
	std::vector<int> init_reader(const char *matrix_file, const char *word_file);
	
	/// @brief Load the similar words vector of the id-th word.
	/// @param[in] id: the index of the word.
	/// @return bool: Success or failure.
	bool load_data(int id);
	

	/// @brief Load the similar words vector of rows (start_id+0, start_id+1, ..., start_id+length-1)
	/// @param[in] start_id: the start index of the word.
	/// @param[in] length: the number of consecutive words we need to load. 
	/// @return bool: Success or failure.
	bool load_data(int start_id, int length);

	/// @brief Release the resource of reader.
	/// @return NULL
	void destroy_reader();
	
	~MatrixFileReader();


	/*
	the word number we need to process.
	*/
	int word_num;
	int*   *r_iptr; //freed
	float* *r_fptr; //freed
private:

	Logger *logger;
	FILE *mat_fp;
	vector<LL> pos;
};


#endif