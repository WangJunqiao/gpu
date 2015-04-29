//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file KMeansClustering.h
/// @brief A virtual class defines some common function of a KMeans clustering algorithm. 
///
/// This is a virtual class. It has two sub classes which implement its function, namely CPU edition and GPU edition.
///
/// @version 1.0
/// @author Junqiao Wang
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////

#ifndef KMEANS_CLUSTERING_H
#define KMEANS_CLUSTERING_H
#include <assert.h>

#include <vector>

#include "../Common/Common.h"
#include "../Common/Logger.h"

using namespace std;

/// @brief A virtual class defines some common function of a KMeans clustering algorithm. 
///
/// This is a virtual class. It has two sub classes which implement its function, namely CPU edition and GPU edition.
class KMeansClustering{
public:
	/// @brief Constructor of a KMeansClustering algorithm.
	/// @param[in] logger: The Logger used to print log.
	/// @return NULL
	explicit KMeansClustering(Logger *logger) {
		this->logger = logger;
	}

	/// @brief some initialize operation before add document.
	/// @param[in] logger: The Logger used to print log.
	/// @return NULL
	virtual void initilize() = 0;


	/// @brief Add a text document into the document pool.
	///
	/// Text document is consist of some English words, separated by white spaces.
	///
	/// @param[in] content: The English document, some white space separated words.
	/// @return NULL
	virtual void add_document(const char *content) = 0;

	/// @brief The main body of clustering algorithm.
	///
	/// Doing the clustering process.
	///
	/// @param[in] k: The number of centroids.
	/// @return NULL
	virtual void run_clustering(int k) = 0;
	
	/// @brief Get the cluster_id-th cluster, represent by document id vector.
	///
	/// @param[in] cluster_id: An integer value in the range [0, k)
	/// @return The cluster_id-th cluster's document ids. 
	vector<int> get_cluster(int cluster_id) {
		assert(cluster_id>=0 && cluster_id<(int)clusters.size());
		return clusters[cluster_id];
	}

	/// @brief Get the cluster, which the doc_id-th document is in.
	///
	/// @param[in] doc_id: the document id you want to check.
	/// @return The cluster's document ids, where the doc_id-th document is in.
	virtual vector<int> get_cluster_which_in(int doc_id) {
		for(int i=0;i<(int)clusters.size();i++){
			bool found = false;
			for(int j=0;j<(int)clusters[i].size();j++) {
				if(clusters[i][j] == doc_id) {
					found = true;
				}
			}
			if(found) return clusters[i];
		}
		return vector<int> (); //doc_id not found
	}

	/// @brief Set the initial situation of k-means clustering algorithm.
	///
	/// @param[in] ids: The document ids you want to set as the initial centroids.
	/// @return NULL
	void set_init_centroids(vector<int> ids) {
		this->init_ids = ids;
	}
	
	/// @brief Release the resource.
	///
	/// @return NULL
	virtual void destroy() = 0;

	/// @brief Print the simple clustering result, used for debugging.
	///
	/// @return NULL
	void print_result() {
		printf("K = %d\n", (int)clusters.size());
		for(int i=0;i<(int)clusters.size();i++) {
			printf("cluster %3d, size = %3d:", i, (int)clusters[i].size());
			for(int j=0;j<(int)clusters[i].size();j++) {
				printf(" %d", clusters[i][j]);
			}
			printf("\n");
		}
	}

	virtual ~KMeansClustering() {};

	vector<vector<int> > clusters;

protected:
	
	vector<int> init_ids;
private:
	Logger *logger;
};

#endif