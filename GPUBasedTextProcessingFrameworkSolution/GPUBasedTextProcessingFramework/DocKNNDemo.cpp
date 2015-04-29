////////////////////////////////////////////////////////////////////
//////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file DocKNNDemo.cpp
/// @brief A document knn demo to demonstrate how to calculate the gpu knn and cpu knn
///
/// This file contains a demo to show how to get the k nearest neighborhood of a document, you
/// can first calculate the tf-idf value according to a corpus dataset, then use the tf-idf value
/// to calculate the k nearest neighborhood of the query documents; or if the tf-idf file is 
/// already exist, then you can directly calculate the k nearest neighborhood, but you should notice
/// that the tf-idf file must be a certain format.
///
/// @version 1.0
/// @author Bowen Liu
/// @date 01.17.2014
///
////////////////////////////////////////////////////////////////////
//////
#include "./DocKNN/DocumentKNN.h"
#include "./DocKNN/MatrixGen.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "./Common/IDFManager.h"
#include "./DocKNN/DataSource.h"

/// @brief a document knn test function
///
/// you can follow the procedure of this function to calculate the k nearest neighborhood of the query documents.
/// this example first calculate the idf value and save it because idf value is kind of stable, then calculate the
/// tf-idf values of corpus and query dataset, then generate the data matrix and calculate the knn.
///
/// @return exit code
int doc_knn_test() {
	//define a log, you can pass a FILE pointer into it or stdout, just like below
	Logger knn_logger(stdout);

	//path of corpus to calculate the tf-idf
	string data_path = "D:/projects/dataset/large/corpus";
	//path of query data to calculate the tf-idf
	string query_path = "D:/projects/dataset/large/query";
	//path of tf-idf value of corpus
	char data_tfidf_path[] = "D:/projects/gpu/source code/GPUBasedTextProcessingFrameworkSolution/GPUBasedTextProcessingFramework/data/data_tf_idf.txt";
	//path of tf-idf value of query data set
	char query_tfidf_path[] = "D:/projects/gpu/source code/GPUBasedTextProcessingFrameworkSolution/GPUBasedTextProcessingFramework/data/query_tf_idf.txt";
	//path of tf-idf value of idf value
	char idf_path[] = "D:/projects/gpu/source code/GPUBasedTextProcessingFrameworkSolution/GPUBasedTextProcessingFramework/data/word_idf.txt";
	//new an object of DataSource to process the corpus
	DocumentSource *corpus = new DataSource(data_path.c_str());
	//new an object of DataSource to process the query data set
	DocumentSource *query = new DataSource(query_path.c_str());
	//define a IDFManager to calculate the tf-idf value
	IDFManager tfidf_calc(&knn_logger);
	//n is the words that we want to retain, they are ordered by the word count, 
	//etc. if n = 100, then the top 100 words and its idf value will be saved
	int n = 5000;
	//calculate idf value
    tfidf_calc.calc_idf(corpus, n, idf_path);
	//calculate tf-idf value, the last parameter is to indicate the idf file is already calculated before
	tfidf_calc.calc_tfidf(corpus, idf_path, data_tfidf_path, HAS_IDF_FILE); //calculate the tfidf of corpus data
	tfidf_calc.calc_tfidf(query, idf_path, query_tfidf_path, HAS_IDF_FILE); //calculate the tfidf of query data
	delete corpus;
	delete query;

	int k = 10;
	char gpu_knn_result[] = "D:/projects/gpu/source code/GPUBasedTextProcessingFrameworkSolution/GPUBasedTextProcessingFramework/data/gpu_knn_result.txt";
	char cpu_knn_result[] = "D:/projects/gpu/source code/GPUBasedTextProcessingFrameworkSolution/GPUBasedTextProcessingFramework/data/cpu_knn_result.txt";
	//define a data matrix object
	DataMatrix data_matrix(&knn_logger);
	//generate matrix of corpus
	data_matrix.generate_matrix(data_tfidf_path);
	//generate matrix of query data set
	data_matrix.generate_query_matrix(query_tfidf_path);
	//define an object of DOCKNN to calculate the k nearest of the query docs
	DOCKNN doc_knn(&knn_logger);
	doc_knn.knn_init(data_matrix, k);
	doc_knn.calc_gpu_knn();
	doc_knn.calc_cpu_knn();
	doc_knn.save_gpu_result(gpu_knn_result);
	doc_knn.save_cpu_result(cpu_knn_result);
	return 0;
}