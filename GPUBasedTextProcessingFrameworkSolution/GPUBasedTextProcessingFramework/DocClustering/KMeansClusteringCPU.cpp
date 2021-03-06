/*
 * Content: kMeans Algorithm using CPU as comparation test
 * Code: 	ycc
 * Time:	summer of 2013
 */

#include "KMeansClusteringCPU.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <cmath>
#include <cctype>
#include <cassert>
#include <algorithm>
#include <vector>
#include <string>

#include "DataPoint.h"

#include "../Common/Common.h"
#include "../Common/Logger.h"

using namespace std;

static int*			g_indices						= NULL;		// assignment of every point
static DataPoint*	g_centroids						= NULL;		// global centroids
static DataPoint*	g_document_vectors				= NULL;		// global document vectors, point to vd[0]
static int			document_counter				= 0;
static int			iteration_counter				= 0;

static vector<DataPoint> vd;

static DataPoint *tmp;

void KMeansClusteringCPU::initilize() {
	srand((unsigned)time(NULL));
	if (g_indices) {
		delete [] g_indices;
		g_indices = NULL;
	}
	if (g_centroids) {
		delete [] g_centroids;
		g_centroids = NULL;
	}
	document_counter = 0;
	iteration_counter = 0;
	vd.clear();
	g_document_vectors = NULL;
	if (tmp) delete tmp;
	tmp = new DataPoint(dimensions); 
	fill(tmp->data, tmp->data + dimensions, 0.0f);
	vd.reserve(110);
}

void KMeansClusteringCPU::destroy() {
	delete [] g_indices;
	g_indices = NULL;
	delete [] g_centroids;
	g_centroids = NULL;
	delete tmp;
}

void KMeansClusteringCPU::add_document(const char *content) {
	vector<int> vn;
	string ts; int wcc = 0;
	while (content[0]) {
		if (isspace(content[0])) {
			if (ts.size() > 0) {
				int id = idf_manager->get_word_id(ts.c_str());
				if (id != -1) {
					++wcc;
					tmp->data[id] += 1;
					vn.push_back(id);
//					fprintf(stderr, "%s inserted\n", ts.c_str());
				}
				ts.clear();
			}
		} else {
			ts.push_back(content[0]);
		}
		++content;
	}
	if (ts.size() > 0) {
		int id = idf_manager->get_word_id((char *)ts.c_str());
		if (id != -1) {
			++wcc;
			tmp->data[id] += 1;
			vn.push_back(id);
//			fprintf(stderr, "%s inserted\n", ts.c_str());
		}
	}
	sort(vn.begin(), vn.end());
	vn.erase(unique(vn.begin(), vn.end()), vn.end());
	for (int i = 0; (size_t)i < vn.size(); ++i)
		tmp->data[vn[i]] *= idf_manager->get_word_idf(vn[i]) / wcc;
//	fprintf(stderr, "%d words\n", wcc);
	vd.push_back(*tmp);
	//LOG(logger, "new_vec");
	for (int i = 0; (size_t)i < vn.size(); ++i) {
		//LOG(logger, " %d:%g", vn[i], tmp->data[vn[i]]);
		tmp->data[vn[i]] = 0;
	}
	g_document_vectors = &vd[0];
	++document_counter;
}

/*
 * funciton: computeCentroids
 * @OutParam centroids		array of centroids
 * @Param X					array of points
 * @Param idx				assignment of every point
 * @Param m					length of points
 * @Param k					#clusters
 */
static void computeCentroids(DataPoint *centroids, DataPoint *X, int *idx, int m, int k, int dimensions) {
	const int n = dimensions;
	int *cc = new int[k];
	fill(cc, cc + k, 0);
	for (int i = 0; i < k; ++i)
		fill(centroids[i].data, centroids[i].data + n, 0.0f);
	for (int i = 0; i < m; ++i) {
		++cc[idx[i]];
		for (int j = 0; j < n; ++j)
			centroids[idx[i]].data[j] += X[i].data[j];
	}
	for (int i = 0; i < k; ++i)
		for (int j = 0; j < n; ++j)
			centroids[i].data[j] /= cc[i];
	delete [] cc;
}

inline float sqr(float x) {
	return x * x;
}

/*
 * function: computeDistance
 * compute squared distance of the two vectors
 */
static float computeDistance2(const DataPoint &x, const DataPoint &y, int dimensions) {
	const int n = dimensions;
	float res = 0;
	for (int i = 0; i < n; ++i)
		res += sqr(x.data[i] - y.data[i]);
	return res;
}

/*
 * funciton: findClosestCentroids
 * @OutParam idx			assignment of every point
 * @Param centroids			array of centroids
 * @Param X					array of points
 * @Param m					length of points
 * @Param k					#clusters
 */
static bool findClosestCentroids(int *idx, DataPoint *centroids, DataPoint *X, int m, int k, int dimensions) {
	const int n = dimensions;
	bool modified = false;
	for (int i = 0; i < m; ++i) {
		int old_idx = idx[i];
		float mind = computeDistance2(X[i], centroids[old_idx], dimensions);
		for (int j = 0; j < k; ++j) {
			if (j == old_idx) continue;
			float dis = computeDistance2(X[i], centroids[j], dimensions);
			if (dis < mind) {
				mind = dis;
				idx[i] = j;
				modified = true;
			}
		}
	}
	return modified;
}

/*
 * function: randomlyInitializeCentroids
 * @OutParam centroids		array of centroids
 * @Param X					array of data vectors
 * @Param k					#clusters
 */
static void randomlyInitializeCentroids(DataPoint *centroids, DataPoint *X, int k) {
	vector<int> tmp(document_counter);
	for (int i = 0; i < document_counter; ++i)
		tmp[i] = i;
	random_shuffle(tmp.begin(), tmp.end());
	for (int i = 0; i < k; ++i) {
		centroids[i] = X[tmp[i]];
	}
}

static void initializeCentroids(DataPoint *centroids, DataPoint *X, vector<int> ids) {
	for(int i=0;i<(int)ids.size();i++) {
		centroids[i] = X[ids[i]];
	}
}

void KMeansClusteringCPU::run_clustering(int k) {
	clock_t ttt = clock();

	LOG(logger, "%s", "KMeansClusteringCPU run_clustering");
	
	assert(document_counter > 0);
	g_indices = new int[document_counter];
	fill(g_indices, g_indices + document_counter, 0);

	g_centroids = new DataPoint[k];
	for(int i=0;i<k;i++) g_centroids[i] = DataPoint(dimensions);

	// TODO: or use other initialization methmod
	if(init_ids.size() == k) {
		LOG(logger, "%s", "kmeans initialize with init_ids");
		initializeCentroids(g_centroids, g_document_vectors, init_ids);
	} else {
		LOG(logger, "%s", "kmeans initialize with random values");
		randomlyInitializeCentroids(g_centroids, g_document_vectors, k);
	}

	while (true) {
		clock_t ch = clock();
		if (!findClosestCentroids(g_indices, g_centroids, g_document_vectors, document_counter, k, dimensions))
			break;
		computeCentroids(g_centroids, g_document_vectors, g_indices, document_counter, k, dimensions);
		ch = clock() - ch;
		++iteration_counter;
		printf("Iteration %3d: %3.6f sec\n", iteration_counter, ch / (double)CLOCKS_PER_SEC);
	}
	clusters = vector<vector<int> >(k);
	for (int i = 0; i < document_counter; ++i)
		clusters[g_indices[i]].push_back(i);
	LOG(logger, "total time used: %lf s\n", (clock()-ttt) / (double)CLOCKS_PER_SEC);
}
// 
// void KMeansClusteringCPU::print_result() {
// 	int k = 0;
// 	puts("results:");
// 	for (int i = 0; i < document_counter; ++i) {
// 		k = max(g_indices[i] + 1, k);
// //		printf("doc %d chooses cluster %d\n", i, g_indices[i]);
// 	}
// 	vector<vector<int> > v(k);
// 	for (int i = 0; i < document_counter; ++i)
// 		v[g_indices[i]].push_back(i);
// 	for (int i = 0; i < k; ++i) {
// 		printf("%dth cluster: %u vector(s):\n", i, v[i].size());
// 		for (int j = 0; j < v[i].size(); ++j)
// 			printf(" %d", v[i][j]);
// 		puts("");
// 	}
// //	printf("%g\n", computeDistance2(g_centroids[1], g_document_vectors[1]));
// 	for (int i = 0; i < k; ++i) {
// 		float a = 0, b = 0;
// 		for (int j = 0; j < document_counter; ++j) {
// 			if (g_indices[j] == i)
// 				a += sqrt(computeDistance2(g_centroids[i], g_document_vectors[j], dimensions));
// 			b += sqrt(computeDistance2(g_centroids[i], g_document_vectors[j], dimensions));
// 		}
// 		a /= v[i].size(); b /= document_counter;
// 		printf("average distance to centroid %3d: %.6f(all), %.6f(belong)\n", i, b, a);
// 	}
// 	//system("pause");
// }
