//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file WordSimCalcCPUImpl.h
/// @brief A word similarity calculator implemented in CPU. 
///
/// This class implement the main process in the CPU.
///
/// @version 1.0
/// @author Junqiao Wang
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////

#ifndef WORD_SIM_CALC_CPU_IMPL_H
#define WORD_SIM_CALC_CPU_IMPL_H

#include "WordSimCalc.h"

/// @brief A word similarity calculator implemented in CPU. 
///
/// This class implement the main process in the CPU.
class CPUWordSimCalculator : public WordSimCalculator {
public:
	/// @brief Constructor of a CPUWordSimCalculator.
	/// @param[in] logger: The Logger used to print log.
	/// @param[in] result_dir: the directory we place the result data.
	/// @param[in] top_words_num: the number of top appeared words we need to analysis.
	/// @return NULL
	CPUWordSimCalculator(Logger *logger, const string &result_dir, int top_words_num);

	void calc_similarity_matrix();
};


#endif

