//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file KMeansClusteringCPU.h
/// @brief This is a subclass of KMeansclustering, which implement all process of clustering in the CPU. 
///
/// This class is used to compare to GPU based KMeansClustering.
///
/// @version 1.0
/// @author Chengchao Yu
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////

#ifndef KMEANS_CLUSTERING_CPU_H
#define KMEANS_CLUSTERING_CPU_H

#include "KMeansClustering.h"

#include "../Common/Logger.h"
#include "../Common/IDFManager.h"

/// @brief This is a subclass of KMeansclustering, which implement all process of clustering in the CPU. 
///
/// This class is used to compare to GPU based KMeansClustering.
class KMeansClusteringCPU : public KMeansClustering {
public:
	/// @brief Constructor of a KMeansClusteringCPU algorithm.
	/// @param[in] logger: The Logger used to print log.
	/// @param[in] idf_manager: An IDFManager pointer we use to retrieve the global idf value of English words. 
	explicit KMeansClusteringCPU(Logger *logger, IDFManager *idf_manager) : KMeansClustering(logger){
		this->logger = logger;
		this->idf_manager = idf_manager;
		dimensions = idf_manager->get_word_num();
	}

	void initilize();

	void add_document(const char *content);

	void run_clustering(int k);

	void destroy();

private:
	Logger *logger;
	int dimensions;
	IDFManager *idf_manager;
};

#endif
