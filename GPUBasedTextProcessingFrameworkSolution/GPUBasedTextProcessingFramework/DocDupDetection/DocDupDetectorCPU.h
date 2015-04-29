//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file DocDupDetectorCPU.h
/// @brief The CPU implementation of document duplicate detector. 
///
/// This class is a subclass of DocDupDetector. It implements all the function in the CPU.
///
/// @version 1.0
/// @author Xinchao Li
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////

#ifndef	DOC_DUP_DETECTOR_CPU_H
#define DOC_DUP_DETECTOR_CPU_H

#include "DocDupDetector.h"

#include <vector>

#include "../Common/Common.h"
#include "../Common/Logger.h"

typedef unsigned u32;
typedef unsigned char uchar;

/// @brief The CPU implementation of document duplicate detector. 
///
/// This class is a subclass of DocDupDetector. It implements all the function in the CPU.
class DocDupDetectorCPU : public DocDupDetector{
public:
	explicit DocDupDetectorCPU(Logger *logger) {
		this->logger = logger;
	}

	void initialize();
	void add_document(string doc);
	void calculate_dups();
	vector<int> get_candidate_dup_docs(int);
	void refine();
	vector<int> get_real_dup_docs(int);
	
	//分片的弱哈希计算函数
	u32 roll_hash(uchar c);
	//对分片计算强哈希值
	u32 sum_hash(uchar c, u32 h);
	//对文档进行分片并合成总的哈希值
	char *spamsum(const uchar *in, int length, int flags, u32 bsize);
	//对两篇文章的哈希值进行大致的比较，若果有公共子串，则有可能是重复文档对,子串长度大于4就可以了。。
	int check_common_substring(const char* hash_value1,const char* hash_value2);

	u32 roll_reset(void);

private:
	Logger *logger;
};

#endif