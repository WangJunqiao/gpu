#include "KMeansClusteringGPU.h"

#include <vector>
#include <set>
#include <map>

#include <cuda_runtime.h>

#include "KMeansClustering.h"

#include "../Common/Common.h"

using namespace std;




static char word[1000000];


void KMeansClusteringGPU::initilize() {
	doc_vecs.clear();
}

void KMeansClusteringGPU::add_document(const char *content) {
	//printf("%s\n", content);
	const char *p = content;
	float *f = (float*)malloc(sizeof(float) * dimensions); //freed
	memset(f, 0, sizeof(float) * dimensions);
	int tot_words = 0, len;
	while(sscanf(p, "%s%n", word, &len) != EOF) {
		p += len;
		int id = idf_manager->get_word_id(word);
		if(id >= 0) {
			f[id] += 1.0f;
			tot_words ++;
		}
	}
	//LOG(logger, "tot words: %d", tot_words);
	for(int i=0;i<dimensions;i++) {
		f[i] = f[i]/tot_words * idf_manager->get_word_idf(i);
	}
	doc_vecs.push_back(f);
}

void KMeansClusteringGPU::random_pick(int k, int dimensions) {
	int doc_num = get_doc_num();
	k = min(k, doc_num);
	set<int> st;
	
	if(init_ids.size() == k) {
		LOG(logger, "%s", "kmeans initialize with init_ids");
		st = set<int>(init_ids.begin(), init_ids.end());
		assert(st.size() == k);
	} else {
		LOG(logger, "%s", "kmeans initialize with random values");
		while((int)st.size() < k) {
			int x = rand() % doc_num;
			st.insert(x);
		}
	}

	//for(int i=0;i<k;i++) st.insert(i);

	printf("random pick centers:");
	for(set<int>::iterator it = st.begin();it!=st.end();++it) {
		printf(" %d", *it);
	} puts("");

	centers.clear();
	for(set<int>::iterator it = st.begin();it!=st.end();++it) {
		float *f = (float*) malloc(sizeof(float) * dimensions); //freed
		memcpy(f, doc_vecs[*it], sizeof(float) * dimensions);
		centers.push_back(f);
	}
}
//利用share memory，每个线程负责一个向量和k个中心点的距离计算
template<int BLOCK_SIZE, int LOAD_SIZE> //(BLOCK_SIZE + 1) * LOAD_SIZE * sizeof(dataType)<= max share memory size
__global__ void calcAllDisByGpu(float **dev_darr, float **dev_karr, int doc_num, int K, int dimensions, float *dev_dis) {
	int tot_threads = blockDim.x * gridDim.x;
	int tid_inblock = threadIdx.x;
	__shared__ float darr[BLOCK_SIZE][LOAD_SIZE];
	__shared__ float karr[LOAD_SIZE];
	for(int row_start = blockDim.x * blockIdx.x;row_start < doc_num;row_start += tot_threads) {
		int row_end = min(row_start + blockDim.x, doc_num);
		for(int col_start = 0; col_start < dimensions; col_start += LOAD_SIZE) {
			int col_end = min(col_start + LOAD_SIZE, dimensions);
			int col_len = col_end - col_start;
			//下面同一个block里面的threads共同复制数据, 利用coalesced memory access
			for(int i = row_start; i < row_end; i ++) {
				for(int j = 0; j < col_len; j += blockDim.x) {
					if(j + tid_inblock + col_start < col_end) 
						darr[i-row_start][j+tid_inblock] = dev_darr[i][j+tid_inblock+col_start];
					__syncthreads();
				}
			}

			for(int k=0;k<K;k++) {
				for(int j = 0; j < col_len; j += blockDim.x) {
					if(j + tid_inblock + col_start < col_end) 
						karr[j+tid_inblock] = dev_karr[k][j+tid_inblock+col_start];
					__syncthreads();
				}
				
				if(tid_inblock < row_end-row_start) {
					float tmp = 0.0f;
					for(int j=0;j<col_len;j++) {
						tmp += (darr[tid_inblock][j] - karr[j]) * (darr[tid_inblock][j] - karr[j]); 
					}
					dev_dis[K*(row_start+tid_inblock) + k] += tmp;
				}
				__syncthreads();
			}
		}
	}
}

//每个线程块负责一个向量和k个中心点的距离计算
template <int BLOCK_SIZE>
__global__ void calcAllDisByGpu2(float **dev_darr, float **dev_karr, int doc_num, int K, int dimensions, float *dev_dis) {
	int tot_blocks = gridDim.x;
	int bid_intotal = blockIdx.x;
	int tid_inblock = threadIdx.x;
	for(int b = bid_intotal;b < doc_num;b += tot_blocks) {
		float* f1 = dev_darr[b];
		for(int k=0;k<K;k++) {
			float *f2 = dev_karr[k];
			__shared__ float sum[BLOCK_SIZE];
			sum[tid_inblock] = 0.0f;
			__syncthreads();
			for(int i=0;i<dimensions;i+=BLOCK_SIZE) {
				if(i+tid_inblock < dimensions) {
					float v1 = f1[i + tid_inblock];
					float v2 = f2[i + tid_inblock];
					sum[tid_inblock] += (v1-v2)*(v1-v2);
				}
				__syncthreads();
			}
			int B = BLOCK_SIZE / 2;
			while(B>=1) {
				if(tid_inblock < B) {
					sum[tid_inblock] += sum[tid_inblock + B];
				}
				__syncthreads();
				B >>= 1;
			}
			if(tid_inblock == 0) {
				dev_dis[b*K + k] = sum[0];
			}
		}
	}
}

//每个线程负责一个向量和k个中心点的距离计算
__global__ void calcAllDisByGpu3(float **dev_darr, float **dev_karr, int doc_num, int K, int dimensions, float *dev_dis) {
	int tot_threads = gridDim.x * blockDim.x;
	int tid_intotal = blockDim.x * blockIdx.x + threadIdx.x;
	for(int b = tid_intotal;b < doc_num;b += tot_threads) {
		float* f1 = dev_darr[b];
		for(int k=0;k<K;k++) {
			float *f2 = dev_karr[k];
			float sum = 0.0f;
			for(int i=0;i<dimensions;i++) {
				float v1 = f1[i];
				float v2 = f2[i];
				sum += (v1-v2)*(v1-v2);
			}
			dev_dis[b*K + k] = sum;
		}
	}
}
//每个线程块负责重计算一个中心点
__global__ void reCalcCenters(float **dev_darr, float **dev_karr, int doc_num, int K, int dimensions, int *dev_belong) {
	int tot_blocks = gridDim.x;
	int block_size = blockDim.x;
	int bid_intotal = blockIdx.x;
	int tid_inblock = threadIdx.x;
	for(int b = bid_intotal;b<K;b+=tot_blocks) {
		float *ka = dev_karr[b];
		for(int i=0;i<dimensions;i+=block_size) {
			if(i+tid_inblock < dimensions) {
				ka[i+tid_inblock] = 0.0f;
			}
			__syncthreads();
		}
		int num = 0;
		for(int d=0;d<doc_num;d++) if(dev_belong[d] == b){
			__syncthreads();
			num ++;
			float *da = dev_darr[d];
			for(int i=0;i<dimensions;i+=block_size) {
				if(i+tid_inblock < dimensions) {
					ka[i+tid_inblock] += da[i+tid_inblock];
				}
				__syncthreads();
			}
		}
		for(int i=0;i<dimensions;i+=block_size) {
			if(i+tid_inblock < dimensions) {
				ka[i+tid_inblock] /= num;
			}
			__syncthreads();
		}
	}
}

void KMeansClusteringGPU::run_clustering(int k) {
	clock_t ttt = clock();

	if(setted) {
		LOG(logger, "Use user defined arguments, block_num = %d, thread_num = %d", block_num, thread_num);
	} else {
		LOG(logger, "Warnning! Use default settings, block_num = %d, thread_num = %d", block_num, thread_num);
	}

	int doc_num = (int)doc_vecs.size();
	k = min(k, doc_num);

	vector<int> cpu_belong(doc_num, 0);

	float** darr = (float**) malloc(sizeof(float*) * doc_num); //freed
	for(int i=0;i<doc_num;i++) {
		float *dev_f;
		//LOG(logger, "i = %d", i);
		safeCudaCall(cudaMalloc(&dev_f, sizeof(float) * dimensions)); //freed
		safeCudaCall(cudaMemcpy(dev_f, doc_vecs[i], sizeof(float)*dimensions, cudaMemcpyHostToDevice));
		darr[i] = dev_f;
	}
	float **dev_darr;
	safeCudaCall(cudaMalloc(&dev_darr, sizeof(float*) * doc_num)); //freed
	safeCudaCall(cudaMemcpy(dev_darr, darr, sizeof(float*) * doc_num, cudaMemcpyHostToDevice));

	random_pick(k, dimensions);
	float** karr = (float**) malloc(sizeof(float*) * k); //freed
	for(int i=0;i<k;i++) {
		float *dev_f;
		safeCudaCall(cudaMalloc(&dev_f, sizeof(float) * dimensions)); //freed
		safeCudaCall(cudaMemcpy(dev_f, centers[i], sizeof(float)*dimensions, cudaMemcpyHostToDevice));
		karr[i] = dev_f;
	}
	float **dev_karr;
	safeCudaCall(cudaMalloc(&dev_karr, sizeof(float*) * k)); //freed
	safeCudaCall(cudaMemcpy(dev_karr, karr, sizeof(float*) * k, cudaMemcpyHostToDevice));

	belong = vector<int>(doc_num, 0);
	float *dis = new float[doc_num * k]; //freed
	float *dev_dis;
	safeCudaCall(cudaMalloc(&dev_dis, sizeof(float) * (doc_num * k))); //freed

	int *dev_belong;
	safeCudaCall(cudaMalloc(&dev_belong, sizeof(int) * doc_num));

	for(int iter = 1;;iter++) {
		safeCudaCall(cudaMemset(dev_dis, 0, sizeof(float) * (doc_num * k)));
		LOG(logger, "%s", "begin cuda call\n");
		int t = clock();
		//calcAllDisByGpu<32, 256> <<<4096, 32>>>(dev_darr, dev_karr, doc_num, k, dimensions, dev_dis);
		calcAllDisByGpu2<128> <<<block_num, 128>>>(dev_darr, dev_karr, doc_num, k, dimensions, dev_dis);
		//calcAllDisByGpu3<<<1024, 32>>>(dev_darr, dev_karr, doc_num, k, dimensions, dev_dis);
		cudaDeviceSynchronize();
		printf("kernel call time: %lf s\n", (clock()-t) / (double)CLOCKS_PER_SEC);
		safeCudaCall(cudaMemcpy(dis, dev_dis, sizeof(float)*(doc_num*k), cudaMemcpyDeviceToHost));
		vector<int> sels;
		for(int i=0;i<doc_num;i++) {
			int id = -1;
			float mi = 1e10;
			float *d = dis + (i*k);
			for(int j=0;j<k;j++) {
				if(d[j] < mi) {
					mi = d[j];
					id = j;
				}
			}
			//printf("%d select %d, mi_dis = %f\n", i, id, mi);
			cpu_belong[i] = id;
			sels.push_back(id);
		}
		if(sels == belong) {
			break;
		}
		belong = sels;

		safeCudaCall(cudaMemcpy(dev_belong, cpu_belong.data(), sizeof(int)*doc_num, cudaMemcpyHostToDevice));
		reCalcCenters<<<block_num, thread_num>>>(dev_darr, dev_karr, doc_num, k, dimensions, dev_belong);
		cudaDeviceSynchronize();

// 		for(int i=0;i<k;i++) {
// 			for(int j=0;j<dimensions;j++) centers[i][j] = 0;
// 		}
// 		vector<int> tot_docs(k, 0);
// 		for(int i=0;i<(int)belong.size();i++) {
// 			int x = belong[i];
// 			for(int j=0;j<dimensions;j++) centers[x][j] += doc_vecs[i][j];
// 			tot_docs[x] ++;
// 		}
// 		for(int i=0;i<k;i++) {
// 			for(int j=0;j<dimensions;j++) {
// 				centers[i][j] /= tot_docs[i];
// 			}
// 		}
// 		for(int i=0;i<k;i++) {
// 			safeCudaCall(cudaMemcpy(karr[i], centers[i], sizeof(float)*dimensions, cudaMemcpyHostToDevice));
// 		}
		LOG(logger, "iterate: %d finished, time used: %d ms\n", iter, clock()-t);
	}

	clusters = vector<vector<int> >(k, vector<int>());
	for(int i=0;i<(int)belong.size();i++) {
		clusters[belong[i]].push_back(i);
	}

	for(int i=0;i<k;i++) {
		safeCudaCall(cudaFree(karr[i]));
	}
	cudaFree(dev_karr);
	free(karr);

	for(int i=0;i<doc_num;i++) {
		safeCudaCall(cudaFree(darr[i]));
	}
	cudaFree(dev_darr);
	free(darr);

	cudaFree(dev_belong);
	cudaFree(dev_dis);
	
	delete[] dis;
	LOG(logger, "total time used: %lf s", (clock()-ttt) / (double)CLOCKS_PER_SEC);
}


// void KMeansClusteringGPU::print_result() {
// 	printf("result:\n");
// 	map<int, vector<int> > mp;
// 	for(int i=0;i<(int)belong.size();i++) {
// 		mp[belong[i]].push_back(i);
// 	}
// 	int c = 1;
// 	for(map<int, vector<int> >::iterator it = mp.begin();it!=mp.end();++it) {
// 		vector<int> v = it->second;
// 		printf("cluster %d: size: %d, docId list:", c++, (int)v.size());
// 		for(int j=0;j<(int)v.size();j++) printf(" %d", v[j]);
// 		puts("");
// 	}
// }

void KMeansClusteringGPU::destroy() {
	for(int i=0;i<(int)doc_vecs.size();i++) {
		//free(original_docs[i]);
		free(doc_vecs[i]);
	}
	for(int i=0;i<(int)centers.size();i++) {
		free(centers[i]);                                                                                                                                                                                                               
	}
}
