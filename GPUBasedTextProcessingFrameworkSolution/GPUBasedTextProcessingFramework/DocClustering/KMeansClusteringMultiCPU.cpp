/*
 * Content: kMeans Algorithm using CPU as comparation test
 * Code: 	ycc
 * Time:	summer of 2013
 */

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
#include <ppl.h>
#include "HashMap.h"
#include "Common.h"
#include "KMeansClustering.h"
using namespace std;

static int*		g_indices						= NULL;		// assignment of every point
static DataPoint*	g_centroids						= NULL;		// global centroids
static DataPoint*	g_document_vectors				= NULL;		// global document vectors, point to vd[0]
static int			document_counter				= 0;
static int			iteration_counter				= 0;
static FILE*		outf							= NULL;		// output file for debug

static vector<DataPoint> vd;

static DataPoint *tmp;

void KMeansClusteringMultiCpu::kmeansInitilize() {
	if (outf) {
		fclose(outf);
	}
	srand((unsigned)time(NULL));
//	srand(526414435);
	outf = fopen("outvec.txt", "w");
	assert(outf);
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
	tmp = new DataPoint; 
	fill(tmp->data, tmp->data + dimensions, 0.0f);
	vd.reserve(110);
}

void KMeansClusteringMultiCpu::kmeansDestroy() {
	fclose(outf);
	delete [] g_indices;
	g_indices = NULL;
	delete [] g_centroids;
	g_centroids = NULL;
	delete tmp;
}

void KMeansClusteringMultiCpu::addDocument(char *content) {
	vector<int> vn;
	string ts; int wcc = 0;
	while (content[0]) {
		if (isspace(content[0])) {
			if (ts.size() > 0) {
				int id = getFeatureId((char *)ts.c_str());
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
		int id = getFeatureId((char *)ts.c_str());
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
		tmp->data[vn[i]] *= idf_values->data[vn[i]] / wcc;
//	fprintf(stderr, "%d words\n", wcc);
	vd.push_back(*tmp);
	fprintf(outf, "new_vec");
	for (int i = 0; (size_t)i < vn.size(); ++i) {
		fprintf(outf, " %d:%g", vn[i], tmp->data[vn[i]]);
		tmp->data[vn[i]] = 0;
	}
	fputc('\n', outf);
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
static void computeCentroids(DataPoint *centroids, DataPoint *X, int *idx, int m, int k) {
	const int n = dimensions;
	int *cc = new int[k];
	fill(cc, cc + k, 0);
	for (int i = 0; i < k; ++i)
		fill(centroids[i].data, centroids[i].data + n, 0.0f);
	// using parallel patterns library (PPL)
	Concurrency::parallel_for(0, m, [=](int i) {	// C++ 11 lambda expression
		++cc[idx[i]];
		for (int j = 0; j < n; ++j)
			centroids[idx[i]].data[j] += X[i].data[j];
	});
	Concurrency::parallel_for(0, k, [=](int i) {
		for (int j = 0; j < n; ++j)
			centroids[i].data[j] /= cc[i];
	});
	/* non parallel version here
	for (int i = 0; i < m; ++i) {
		++cc[idx[i]];
		for (int j = 0; j < n; ++j)
			centroids[idx[i]].data[j] += X[i].data[j];
	}
	for (int i = 0; i < k; ++i)
		for (int j = 0; j < n; ++j)
			centroids[i].data[j] /= cc[i];
	*/
	delete [] cc;
}

inline float sqr(float x) {
	return x * x;
}

/*
 * function: computeDistance
 * compute squared distance of the two vectors
 */
static float computeDistance2(const DataPoint &x, const DataPoint &y) {
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
static bool findClosestCentroids(int *idx, DataPoint *centroids, DataPoint *X, int m, int k) {
	const int n = dimensions;
	bool modified = false;
	// using parallel patterns library (PPL)
	Concurrency::parallel_for(0, m, [&modified, idx, X, centroids, k] (int i) {
													// C++ 11 lambda expression
		int old_idx = idx[i];
		float mind = computeDistance2(X[i], centroids[old_idx]);
		for (int j = 0; j < k; ++j) {
			if (j == old_idx) continue;
			float dis = computeDistance2(X[i], centroids[j]);
			if (dis < mind) {
				mind = dis;
				idx[i] = j;
				modified = true;
			}
		}
	});
	/* non parallel version here
	for (int i = 0; i < m; ++i) {
		int old_idx = idx[i];
		float mind = computeDistance2(X[i], centroids[old_idx]);
		for (int j = 0; j < k; ++j) {
			if (j == old_idx) continue;
			float dis = computeDistance2(X[i], centroids[j]);
			if (dis < mind) {
				mind = dis;
				idx[i] = j;
				modified = true;
			}
		}
	}
	*/
	return modified;
}

/*
 * function: randomlyInitializeCentroids
 * @OutParam centroids		array of centroids
 * @Param X					array of data vectors
 * @Param k					#clusters
 */
static void randomlyInitializeCentroids(DataPoint *centroids, DataPoint *X, int k) {
	int *tmp = new int[document_counter];
	for (int i = 0; i < document_counter; ++i)
		tmp[i] = i;
	random_shuffle(tmp, tmp + document_counter);
	for (int i = 0; i < k; ++i) {
		printf("%d th vector chosen.\n", tmp[i]);
		centroids[i] = X[tmp[i]];
//		for (int j = 0; j < dimensions; ++j)
//			if (abs(X[tmp[i]].data[j]) > 1e-9)
//				printf(" %d:%g", j, X[tmp[i]].data[j]);
//		puts("");
	}
	delete [] tmp;
}

void KMeansClusteringMultiCpu::kmeansClustering(int k) {
	assert(document_counter > 0);
	g_indices = new int[document_counter];
	fill(g_indices, g_indices + document_counter, 0);
	g_centroids = new DataPoint[k];
	// TODO: or use other initialization methmod
	randomlyInitializeCentroids(g_centroids, g_document_vectors, k);

	while (true) {
		clock_t ch = clock();
		if (!findClosestCentroids(g_indices, g_centroids, g_document_vectors, document_counter, k))
			break;
		computeCentroids(g_centroids, g_document_vectors, g_indices, document_counter, k);
		ch = clock() - ch;
		++iteration_counter;
		printf("Iteration %3d: %3.6f sec\n", iteration_counter, ch / (double)CLOCKS_PER_SEC);
	}
	puts("done!");
}

void KMeansClusteringMultiCpu::printResult() {
	int k = 0;
	puts("results:");
	for (int i = 0; i < document_counter; ++i) {
		k = max(g_indices[i] + 1, k);
//		printf("doc %d chooses cluster %d\n", i, g_indices[i]);
	}
	vector<vector<int> > v(k);
	for (int i = 0; i < document_counter; ++i)
		v[g_indices[i]].push_back(i);
	for (int i = 0; i < k; ++i) {
		printf("%dth cluster: %u vector(s):\n", i, v[i].size());
		for (int j = 0; j < v[i].size(); ++j)
			printf(" %d", v[i][j]);
		puts("");
	}
//	printf("%g\n", computeDistance2(g_centroids[1], g_document_vectors[1]));
	for (int i = 0; i < k; ++i) {
		float a = 0, b = 0;
		for (int j = 0; j < document_counter; ++j) {
			if (g_indices[j] == i)
				a += sqrt(computeDistance2(g_centroids[i], g_document_vectors[j]));
			b += sqrt(computeDistance2(g_centroids[i], g_document_vectors[j]));
		}
		a /= v[i].size(); b /= document_counter;
		printf("average distance to centroid %3d: %.6f(all), %.6f(belong)\n", i, b, a);
	}
	//system("pause");
}