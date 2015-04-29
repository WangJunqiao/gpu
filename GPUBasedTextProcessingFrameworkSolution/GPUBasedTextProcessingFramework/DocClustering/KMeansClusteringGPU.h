//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file KMeansClusteringGPU.h
/// @brief This is a subclass of KMeansclustering, which implement main process (time consuming part) of clustering in the GPU. 
///
/// This class is main class represent to you to use in your own project.
///
/// @version 1.0
/// @author Junqiao Wang
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////

#ifndef KMEANS_CLUSTERING_GPU_H
#define KMEANS_CLUSTERING_GPU_H

#include "KMeansClustering.h"
#include "../Common/IDFManager.h"

/// @brief This is a subclass of KMeansclustering, which implement main process (time consuming part) of clustering in the GPU. 
///
/// This class is main class represent to you to use in your own project.
class KMeansClusteringGPU : public KMeansClustering {
public:
	/// @brief Constructor of a KMeansClusteringGPU algorithm.
	/// @param[in] logger: The Logger used to print log.
	/// @param[in] idf_manager: An IDFManager pointer we use to retrieve the global idf value of English words. 
	explicit KMeansClusteringGPU(Logger *logger, IDFManager *idf_manager) : KMeansClustering(logger) {
		this->logger = logger;
		this->idf_manager = idf_manager;
		dimensions = idf_manager->get_word_num();
		block_num = 2048;
		thread_num = 128;
		setted = 0;
		LOG(logger, "dimensions = %d", dimensions);
	}
	/// @brief Set the run time parameters of GPU, namely the block number and thread number per block.
	/// @param[in] block_num: the block number while run clustering.
	/// @param[in] thread_num: the thread number per block.
	/// @return NULL
	void set_params(int block_num, int thread_num) {
		this->block_num = block_num;
		this->thread_num = thread_num;
		setted = 1;
	}
	
	/// @brief Get the current documents in the document pool.
	/// @return An integer represent the number of documents.
	int get_doc_num() {
		return (int)doc_vecs.size();
	}

	void initilize();

	void add_document(const char *content);

	void run_clustering(int k);

	void destroy();

private:
	void random_pick(int k, int dimensions);

	Logger *logger;
	int dimensions;
	IDFManager *idf_manager;
	int block_num, thread_num, setted;
	vector<float*> doc_vecs;
	vector<float*> centers;
	vector<int> belong;
};


#endif