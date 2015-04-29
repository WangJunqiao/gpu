////////////////////////////////////////////////////////////////////
//////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file DocumentKNN.h
/// @brief A @ref DOCKNN class to compute the nearest neighborhoods of a given query
///
/// This file contains a @ref DOCKNN class, you can use @ref calc_gpu_knn method to
/// compute the k nearest neighborhoods using gpu, or @ref calc_cpu_knn method to
/// compute the k nearest neighborhoods using cpu.
///
/// @version 1.0
/// @author Bowen Liu
/// @date 01.17.2014
///
////////////////////////////////////////////////////////////////////
//////
#ifndef _GPUKNN_H_
#define _GPUKNN_H_

#include <vector>
#include <ctime>
#include "MatrixGen.h"
#include "../Common/Logger.h"
using namespace std;

/// @brief A DOCKNN class to calculate k nearest neighborhood both use GPU and CPU
///
/// The objects of this class not only can calculate the k nearest neighborhood using
/// CPU but also using GPU
/// sample use: DOCKNN *doc_knn = new DOCKNN(&logger);
///				doc_knn->knn_init(data_matrix, k);
///				doc_knn->calc_gpu_knn();
///				doc_knn->calc_cpu_knn();
///				doc_knn->print_gpu_result();
///				doc_knn->print_gpu_result();
///	or you just can save the result using @ref save_gpu_result and save_cpu_result
/// @note 1. you must first use the @ref doc_knn->knn_init method to initialize
class DOCKNN {
public:
	/// @brief construct method
	///
	/// the construct method is an explicit one, so you must pass a pointer of Logger object
	/// into the construct method.
	///
	/// @param[in] logger: the Logger pointer to print log info.
	explicit DOCKNN(Logger *logger) {
		this->logger = logger;
	}

	/// @brief initial method to initialize
	///
	/// before you use the object of this class you should use this method to initialize the 
	/// object.
	///
	/// @param[in] _m: DataMatrix object contains the data and query data.
	/// @param[in] _k: the number of nearest neighborhood
	/// @return NULL
	void knn_init(const DataMatrix &_m, const int &_k){
		m = _m;
		k = _k;
	}

	/// @brief calculate the neighborhood using GPU 
	///
	/// after you initialize the object you can call this method to calculate the neighborhood 
	///
	/// @return NULL
	void calc_gpu_knn();

	/// @brief calculate the neighborhood using CPU 
	///
	/// after you initialize the object you can call this method to calculate the neighborhood 
	///
	/// @return NULL
	void calc_cpu_knn();

	/// @brief print the GPU KNN result 
	///
	/// after you calculate the k nearest neighborhood using @ref calc_gpu_knn you can use this method 
	/// to print the GPU result to the standard output 
	///
	/// @return NULL
	void print_gpu_result();

	/// @brief print the CPU KNN result
	///
	/// after you calculate the k nearest neighborhood using @ref calc_cpu_knn you can use this method 
	/// to print the CPU result to the standard output
	///
	/// @return NULL
	void print_cpu_result();

	/// @brief save the gpu result to file
	///
	/// after you calculate the k nearest neighborhood using @ref calc_gpu_knn you can use this method 
	/// to save the GPU result to file, the file name is given by the @param gpu_path 
	///
	/// @param[in] gpu_path: the file path you want to save the GPU result to.
	/// @return NULL
	void save_gpu_result(char *gpu_path);

	/// @brief save the cpu result to file
	///
	/// after you calculate the k nearest neighborhood using @ref calc_cpu_knn you can use this method 
	/// to save the CPU result to file, the file name is given by the @param cpu_path 
	///
	/// @param[in] cpu_path: the file path you want to save the CPU result to.
	/// @return NULL
	void save_cpu_result(char *cpu_path);
	~DOCKNN(){}
private:
	DataMatrix m;
	vector<pair<int, double> > gpu_result, cpu_result;
	int k;
	Logger *logger;
};

#endif