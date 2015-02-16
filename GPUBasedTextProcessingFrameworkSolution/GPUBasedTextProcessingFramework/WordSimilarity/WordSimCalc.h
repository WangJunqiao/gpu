//////////////////////////////////////////////////////////////////////////
//////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file WordSimCalc.h
/// @brief A virtual class defines some common function of a Word Similarity Calculation.
///
/// This is a virtual class. It has two sub classes which implement its function, namely CPU edition and GPU edition.
///
/// @version 1.0
/// @author Junqiao Wang
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////
//////
#ifndef WORDSIMCALC_H
#define WORDSIMCALC_H

#include "../DataSource/DocumentSource.h"
#include "../Common/Logger.h"

#include <time.h>

/// @brief A virtual class defines some common function of a Word Similarity Calculation.
///
/// This is a virtual class. It has two sub classes which implement its function, namely CPU edition and GPU edition.
class WordSimCalculator{
public:
	/// @brief Constructor of a WordSimCalculator.
	/// @param[in] logger: The Logger used to print log.
	/// @param[in] result_dir: the directory we place the result data.
	/// @param[in] top_words_num: the number of top appeared words we need to analysis.
	/// @return NULL
	WordSimCalculator(Logger *logger, const string &result_dir, int top_words_num);

	/// @brief Calculate the first order of the word similarity matrix.
	/// @param[in] doc_src: The document source we will use to calculate the first order matrix. In order to ensure the 
	/// accuracy, the document source needs to be as big as possible. The bigger the better.
	
	/*
	According to the document source, calculate the first order matrix and list the top ranked words.
	
	matrix file(binary file) is a compressed representation of word similarity matrix, its structure like this:
	num_of_sim_words_0(int, m) sum_of_sim_0(float) id_of_word_0(int) sim_of_word_0(float) ... id_of_word_m-1(int) sim_of_word_m-1(float)
	...
	num_of_sim_words_n(int, x) sum_of_sim_n(float) id_of_word_0(int) sim_of_word_o(float) ... id_of_word_x-1(int) sim_of_word_x-1(float)
	
	word file(text file) is a list of all top ranked words according to the big corpus(from doc_src), its structure like this:
	row1: (1)word_id (2)word_str (3)start_pos_of_simlist_in_matrixfile (4)num_of_sim_words
	row2: (1)word_id (2)word_str (3)start_pos_of_simlist_in_matrixfile (4)num_of_sim_words
	...
	rown: (1)word_id (2)word_str (3)start_pos_of_simlist_in_matrixfile (4)num_of_sim_words

	for example(matrix file):
	1 0.3 2 0.3
	1 0.4 2 0.4
	2 0.7 0 0.3 1 0.4
	the real word similarity matrix is
	|0.0 0.0 0.3|
	|0.0 0.0 0.4|
	|0.3 0.4 0.0|

	for example(word file):
	0 how 0  1
	1 who 16 1
	2 she 32 2
	*/
	/// @return NULL
	void calc_mutual_info_matrix(DocumentSource *doc_src, int win_size);
	/// @brief According to first order matrix, calculate the second order similarity matrix.
	/// @param[in] order: The current order of the matrix. 
	/// For example, order = 1, then the structure of matrix_file2 is same as matrix_file1.
	/// @return NULL
	virtual void calc_similarity_matrix() = 0;

	clock_t core_time;
	
protected:
	/*
	把一个triples的文件整合成一个matrix_file, 数据存放形式的转换
	*/
	void rebuild_triples(int order1, int order2);

	/*
	Get the word file name.
	*/
	string get_word_file_name();
	/*
	Get the matirx file name from order.
	*/
	string get_matrix_file_name(int order);

	Logger *logger;
	int top_words_num;
	string result_dir;
private:
	void find_top_words(DocumentSource *doc_src);
	DISALLOW_COPY_AND_ASSIGN(WordSimCalculator);
};

#endif


