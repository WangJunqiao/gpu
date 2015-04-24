//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file WordSimCalcGPUImpl.h
/// @brief A word similarity calculator implemented in GPU. 
///
/// This class implement the main process in the GPU in order to accelerate the calculation.
///
/// @version 1.0
/// @author Junqiao Wang
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////

#ifndef WORDSIMCALCGPUIMPL_H
#define WORDSIMCALCGPUIMPL_H

#include "WordSimCalc.h"

/// @brief A word similarity calculator implemented in GPU. 
///
/// This class implement the main process in the GPU in order to accelerate the calculation.
class GPUWordSimCalculator : public WordSimCalculator {
public:
	/// @brief Constructor of a GPUWordSimCalculator.
	/// @param[in] logger: The Logger used to print log.
	/// @param[in] result_dir: the directory we place the result data.
	/// @param[in] top_words_num: the number of top appeared words we need to analysis.
	/// @return NULL
	GPUWordSimCalculator(Logger *logger, const string &root_dir, const string &result_dir, int top_words_num, int win_size);

	void calc_similarity_matrix();

	/// @brief Set the parameters of GPUWordSimCalculator.
	/// @param[in] block_num: The block number.
	/// @param[in] thread_num: The thread number in each block.
	/// @param[in] pairs_limit: The number of pairs we will process in batch.
	/// @return NULL
	void set_params(int block_num, int thread_num, int pairs_limit);
private:
	int block_num, thread_num, pairs_limit, setted;
};

#endif