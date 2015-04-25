//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file DocDupDetectorGPU.h
/// @brief The GPU implementation of document duplicate detector. 
///
/// This class is a subclass of DocDupDetector. It implements all the function in the GPU.
///
/// @version 1.0
/// @author Junqiao Wang
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////

#ifndef DOC_DUP_DETECTOR_GPU_H
#define DOC_DUP_DETECTOR_GPU_H

#include "DocDupDetector.h"

#include "../Common/Common.h"
#include "../Common/Logger.h"

/// @brief The GPU implementation of document duplicate detector. 
///
/// This class is a subclass of DocDupDetector. It implements all the function in the GPU.
class DocDupDetectorGPU : public DocDupDetector {
public:
	explicit DocDupDetectorGPU(Logger *logger) {
		this->logger = logger;
	}

	void initialize();
	void add_document(string doc);
	void calculate_dups();
	vector<int> get_candidate_dup_docs(int doc_id);
	void refine();
	vector<int> get_real_dup_docs(int doc_id);

	void set_param(int blocks, int threads, int methods);

	~DocDupDetectorGPU();

private:
	void useMethod1(int doc_num, char **d_hashstrs, int *d_hashstrs_length, int *d_startId, int *d_endedId);
	void useMethod3(int doc_num, char **d_hashstrs, int *d_hashstrs_length, int *d_startId, int *d_endedId);
	Logger *logger;
	int blocks, threads, method;
};

#endif