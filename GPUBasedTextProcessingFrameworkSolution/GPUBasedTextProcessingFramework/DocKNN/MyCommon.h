////////////////////////////////////////////////////////////////////
//////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file MyCommon.h
/// @brief A file to define common things
///
/// This file contains some common things in the document knn project 
///
/// @version 1.0
/// @author Bowen Liu
/// @date 01.17.2014
///
////////////////////////////////////////////////////////////////////
#ifndef _MY_COMMON_H_
#define _MY_COMMON_H_
#include <string>
#include <math.h>
using namespace std;

/// @brief A macro that define the block size when using GPU to calculate
#define BLOCK_SIZE 16

/// @brief A type define to give the type of data
typedef double DataType;



#endif