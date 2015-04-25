#include "DocDupDetectorGPU.h"

#include <cstring>
#include <iostream>
#include <algorithm>
#include <cassert>
#include <ctime>
#include <queue>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include "../Common/Common.h"
#include "../Common/CudaMemoryManager.cuh"

using namespace std;

#define SHARED_MEMORY_SIZE 49152
#define MAX_DOCUMENTS 200010

#define MAX_HASH_STR_LEN 128

#define ROLLING_WINDOW 7
#define THRESHOLD 0.7
#define MINIMUM_SCORE 80
#define BASE 130
#define ALPHA_NUMBER 94



#define INF 0x6fff

typedef unsigned char EditDistT;
typedef unsigned int hash_type;


static vector <char*> contents_buffer;
static vector <char*> hashstrs_buffer;
static vector <int>   contents_length;
static vector <int>   hashstrs_length;
//以上是original data

static vector<int> candies[MAX_DOCUMENTS];
static vector<int> real_dups[MAX_DOCUMENTS];
//以上是答案

static vector<pair<int, int> > order_by_length;

static double average_len;
void DocDupDetectorGPU::initialize() {
	contents_buffer.clear();
	hashstrs_buffer.clear();
	average_len = 0.0;
	for(int i=0;i<MAX_DOCUMENTS;i++) {
		candies[i].clear();
		real_dups[i].clear();
	}
	blocks = threads = 64; //default value
	method = -1; //unset
}

void DocDupDetectorGPU::set_param(int blocks, int threads, int method) {
	this->blocks = blocks;
	this->threads = threads;
	this->method = method;
}

void DocDupDetectorGPU::add_document(string doc) {
	//printf("doc length: %d, %s\n", doc.length(), doc.c_str());
	hash_type Tmp = 1;
	for(int i=0;i<ROLLING_WINDOW;i++) Tmp *= BASE;

	char *p = new char[doc.length() + 1];
	strcpy(p, doc.c_str());

	string code = "";
	int block_size = 4;
	while(code=="" || code.length() >= MAX_HASH_STR_LEN) {
		code = "";
		for(int i=0, j;i<(int)doc.length();i++) {
			unsigned roll_h = 0, h = 0;
			for(j=i;j<(int)doc.length();j++) {
				h = h * BASE + doc[j];
				roll_h = roll_h * BASE + doc[j];
				if(j-i >= ROLLING_WINDOW) {
					roll_h -= Tmp * doc[j - ROLLING_WINDOW];
				}
				if(roll_h % block_size == block_size - 1) {
					break;
				}
			}
			code += (char)('!' + h % ALPHA_NUMBER);
			i = j;
		}
		block_size *= 2;
	}
	//cout<<block_size<<endl;

	char *h = new char[code.length()+1];
	strcpy(h, code.c_str());

	//cout<<code<<endl;
	LOG(logger, "doc_id = %d, hash_value = %s", (int)contents_buffer.size(), code.c_str());
	average_len += code.length();

	contents_buffer.push_back(p);
	contents_length.push_back(strlen(p));
	hashstrs_buffer.push_back(h);
	hashstrs_length.push_back(strlen(h));
}

//计算[b1, b2)内的所有串跟其他串的重复情况, 一般情况下tot_blocks = b2 - b1
__global__ void calcDupsByGpu(char **d_hashstrs, int *d_hashstrs_length, int *d_startId, int *d_endedId, int b1, int b2, int doc_num, char *char_map) {
	//int tot_blocks = gridDim.x;
	//assert(blockDim.x == 1);
	int block_id = blockIdx.x;
	
	int threads = blockDim.x;
	int thread_id = threadIdx.x;
	if(b1 + block_id >= b2) 
		return;

	int start = d_startId[b1 + block_id];
	int ended = d_endedId[b1 + block_id];



	__shared__ EditDistT edit_dis_buf[2 * MAX_HASH_STR_LEN * 95];
	__shared__ char str1_buf[MAX_HASH_STR_LEN * 95];
	__shared__ char str2_buf[MAX_HASH_STR_LEN * 95];
	EditDistT* edit_dis[2];
	char *str1, *str2;
	edit_dis[0] = edit_dis_buf + thread_id * MAX_HASH_STR_LEN * 2;
	edit_dis[1] = edit_dis[0] + MAX_HASH_STR_LEN;
	str1 = str1_buf + thread_id * MAX_HASH_STR_LEN;
	str2 = str2_buf + thread_id * MAX_HASH_STR_LEN;

	int len1 = d_hashstrs_length[b1 + block_id], len2;
	//cudaMemcpy(str1+1, d_hashstrs[b1+bid_intotal], sizeof(char) * len1, cudaMemcpyDeviceToDevice);
	for(int i=0;i<len1;i++) {
		str1[i+1] = d_hashstrs[b1 + block_id][i];
	}
	for(int to = start + thread_id;to <= ended;to += threads) {
		len2 = d_hashstrs_length[to];
		//cudaMemcpy(str2+1, d_hashstrs[to], sizeof(char) * len2, cudaMemcpyDeviceToDevice);
		for(int i=0;i<len2;i++) {
			str2[i+1] = d_hashstrs[to][i];
		}
		int now = 0;
		for(int j=0;j<=len2;j++) {
			edit_dis[now][j] = j;
		}
		for(int i=1;i<=len1;i++) {
			for(int j=0;j<=len2;j++) {
				edit_dis[!now][j] = INF;
			}
			for(int j=1;j<=len2;j++) {
				edit_dis[!now][j] = (edit_dis[now][j] < edit_dis[!now][j-1] ? edit_dis[now][j] : edit_dis[!now][j-1]) + 1;
				if(edit_dis[now][j-1]+1 < edit_dis[!now][j])
					edit_dis[!now][j] = edit_dis[now][j-1] + 1;
				if(str1[i] == str2[j] && edit_dis[now][j-1] < edit_dis[!now][j])
					edit_dis[!now][j] = edit_dis[now][j-1];
			}
			now = !now;
		}
		//printf("edit-dis[%d - %d] = %d\n", b1 + bid_intotal, to, edit_dis[now][len2]);
		if(edit_dis[now][len2] < len1 * (1.0 - THRESHOLD) &&
			edit_dis[now][len2] < len2 * (1.0 - THRESHOLD)) {
				char_map[block_id * doc_num + to] = 'y';
				//printf("debug: %d - %d\n", b1 + bid_intotal, to);
		}
	}
}

template <typename T> __device__ T gmin(T a, T b) {
	return a < b ? a : b;
}

#define checkmin(a, b) if((a)>(b))a=b


template <typename T> __device__ T gmax(T a, T b) {
	return a > b ? a : b;
}


//一个block负责计算一个str和其他str的编辑距离。
__global__ void calcDupsByGpu3(char **d_hashstrs, int *d_hashstrs_length, int *d_startId, int *d_endedId, int b1, int b2, int doc_num, char *char_map, EditDistT *edit_dis) {
	int totBlocks = gridDim.x;
	int nThreadPerBlock = blockDim.x;
	int blockId = blockIdx.x;
	int threadId = threadIdx.x;

	if(b1 + blockId >= b2) return;

	int tid_intotal = blockId * nThreadPerBlock + threadId;
	EditDistT* dp[2];
	dp[0] = edit_dis + tid_intotal * 2 * MAX_HASH_STR_LEN;
	dp[1] = dp[0] + MAX_HASH_STR_LEN;

	__shared__ char str2[MAX_HASH_STR_LEN];
	int len2 = d_hashstrs_length[b1+blockId], len1;
	if(threadId == 0) {
		for(int i=0;i<len2;i++) {
			str2[i+1] = d_hashstrs[b1+blockId][i];
		}
	}
	int start = d_startId[b1 + blockId];
	int ended = d_endedId[b1 + blockId];
	
	for(int ite = 0;;ite++) {
		int to = start + ite * nThreadPerBlock + threadId;
		if(to > ended) break;
		len1 = d_hashstrs_length[to];
		//cudaMemcpy(str1+1, d_hashstrs[to], sizeof(char) * len1, cudaMemcpyDeviceToDevice);
		char *str1 = d_hashstrs[to];

		int now = 0;
		for(int j=0;j<=len2;j++) {
			dp[now][j] = j;
		}
		for(int i=1;i<=len1;i++) {
			char ch = str1[i-1];
			for(int j=0;j<=len2;j++) {
				dp[!now][j] = INF;
			}
			EditDistT a, b, c, d;
			for(int j=1;j<=len2;j++) {
				b = dp[now][j];
				c = dp[!now][j-1];
				d = dp[now][j-1];
				a = (b < c ? b : c) + 1;
				if(d+1 < a)
					a = d + 1;
				if(str2[j] == ch && d < a)
					a = d;
				dp[!now][j] = a;
			}
			now = !now;
		}
		//printf("edit-dis[%d - %d] = %d\n", b1 + bid_intotal, to, dp[now][len1]);
		if(dp[now][len2] < len2 * (1.0 - THRESHOLD) &&
			dp[now][len2] < len1 * (1.0 - THRESHOLD)) {
				char_map[blockId * doc_num + to] = 'y';
				//printf("debug: %d - %d\n", b1 + bid_intotal, to);
		}
	}
}

static char *  h_contents[MAX_DOCUMENTS];
static int     h_contents_length[MAX_DOCUMENTS]; //内容

static char *  h_hashstrs[MAX_DOCUMENTS];
static int     h_hashstrs_length[MAX_DOCUMENTS]; //哈希串
//以上是经过order_by_length转换之后的data, 值都是GPU中的地址

//static int h_ans_len[MAX_BLOCKS];
//static int h_ans[MAX_DOCUMENTS];

//以上是临时数据

static int startId[MAX_DOCUMENTS];
static int endedId[MAX_DOCUMENTS];

static CudaMemoryManager<char> memo_mana_c;
static CudaMemoryManager<int>  memo_mana_i;
static CudaMemoryManager<EditDistT> memo_mana_s;


void DocDupDetectorGPU::useMethod1(int doc_num, char **d_hashstrs, int *d_hashstrs_length, int *d_startId, int *d_endedId) {
	int max_threads = SHARED_MEMORY_SIZE / (4 * sizeof(EditDistT) * MAX_HASH_STR_LEN) - 1;
	if (threads > max_threads) {
		this->threads = max_threads;
		LOG(logger, "set threads value is too big, max values is %d", max_threads);
	}
	LOG(logger, "blocks = %d, threads = %d", blocks, threads);
	clock_t ttt = clock();

	char *char_map = memo_mana_c.gpu_malloc(blocks * doc_num);
	char *h_char_map = (char*)malloc(blocks * doc_num); //freed

	for(int b1 = 0; b1 < doc_num; b1 += blocks) {
		int b2 = min(b1 + blocks, doc_num);
		LOG(logger, "Processing docs[%6d, %6d)......", b1, b2);
		int t = clock();
		safeCudaCall(cudaMemset(char_map, 0, sizeof(char)*(blocks * doc_num)));
		calcDupsByGpu<<<blocks, threads>>>(d_hashstrs, d_hashstrs_length, d_startId, d_endedId, b1, b2, doc_num, char_map);
		cudaDeviceSynchronize();
		LOG(logger, "time used: %lf s\n", (clock()-t) / (double)CLOCKS_PER_SEC);

		t = clock();
		safeCudaCall(cudaMemcpy(h_char_map, char_map, sizeof(char) * blocks * doc_num, cudaMemcpyDeviceToHost));
		for(int i = 0;i < b2 - b1; i ++) {
			for(int j = 0;j < doc_num; j ++) if (h_char_map[i * doc_num + j] == 'y') {
				int id1 = order_by_length[b1 + i].second;
				int id2 = order_by_length[j].second;
				if(id1 == id2) continue;
				candies[id1].push_back(id2);
				if(id1 > id2) candies[id2].push_back(id1);
			}
		}
		LOG(logger, "data copy and insert: %lf s\n", (clock()-t) / (double)CLOCKS_PER_SEC);
	}
	free(h_char_map);
	LOG(logger, "calculateDups time: %lf s\n", (clock()-ttt) / (double)CLOCKS_PER_SEC);
}

void DocDupDetectorGPU::useMethod3(int doc_num, char **d_hashstrs, int *d_hashstrs_length, int *d_startId, int *d_endedId) {
	clock_t ttt = clock();
	LOG(logger, "blocks = %d, threads = %d", blocks, threads);

	EditDistT *edit_dis = memo_mana_s.gpu_malloc(blocks * threads * 2 * MAX_HASH_STR_LEN);

	char *char_map = memo_mana_c.gpu_malloc(blocks * doc_num);
	char *h_char_map = (char*)malloc(blocks * doc_num); //freed

	for(int b1 = 0; b1 < doc_num; b1 += blocks) {
		int b2 = min(b1 + blocks, doc_num);
		LOG(logger, "Processing docs[%6d, %6d)......", b1, b2);
		int t = clock();
		safeCudaCall(cudaMemset(char_map, 0, sizeof(char)*(blocks * doc_num)));
		calcDupsByGpu3<<<blocks, threads>>>(d_hashstrs, d_hashstrs_length, d_startId, d_endedId, b1, b2, doc_num, char_map, edit_dis);
		cudaDeviceSynchronize();
		LOG(logger, "time used: %lf s", (clock()-t) / (double)CLOCKS_PER_SEC);

		t = clock();
		safeCudaCall(cudaMemcpy(h_char_map, char_map, sizeof(char) * blocks * doc_num, cudaMemcpyDeviceToHost));
		for(int i = 0;i < b2 - b1; i ++) {
			for(int j = 0;j < doc_num; j ++) if (h_char_map[i * doc_num + j] == 'y') {
				int id1 = order_by_length[b1 + i].second;
				int id2 = order_by_length[j].second;
				if(id1 == id2) continue;
				candies[id1].push_back(id2);
				if(id1 > id2) candies[id2].push_back(id1);
			}
		}
		LOG(logger, "data copy and insert: %lf s", (clock()-t) / (double)CLOCKS_PER_SEC);
	}
	free(h_char_map);
	LOG(logger, "calculateDups time: %lf s", (clock()-ttt) / (double)CLOCKS_PER_SEC);
}


void DocDupDetectorGPU::calculate_dups() {
	LOG(logger, "%s", "Begin calculate doc dups by GPU.");
	LOG(logger, "average_len = %lf", average_len / contents_buffer.size());
	if (MAX_HASH_STR_LEN >= 256) {
		LOG(logger, "%s", "MAX_HASH_STR_LEN bigger than 255, edit distance value type is unsigned char, not support!!!");
		exit(0);
	}
	int ttt = clock();
	int doc_num = contents_buffer.size();
	order_by_length.clear();
	double sumL = 0.0, maxL = 0.0;
	for(int i=0;i<doc_num;i++) {
		order_by_length.push_back(make_pair(hashstrs_length[i], i));
		sumL += hashstrs_length[i];
		maxL = max(maxL, (double)hashstrs_length[i]);
	}
	sort(order_by_length.begin(), order_by_length.end());

	//copy the document content into gpu memory.
	int maxLD = 0;
// 	for(int i=0;i<doc_num;i++) {
// 		int id = order_by_length[i].second;
// 		char *d_c = memo_mana_c.gpu_malloc(contents_length[id]);
// 		safeCudaCall(cudaMemcpy(d_c, contents_buffer[id], sizeof(char)*contents_length[id], cudaMemcpyHostToDevice));
// 		h_contents[i] = d_c;
// 		h_contents_length[i] = contents_length[id];
// 		maxLD = max(maxLD, h_contents_length[i]);
// 	}
// 	char **d_contents = memo_mana_c.copy_bucks(h_contents, doc_num);


	for(int i=0;i<doc_num;i++) {
		int id = order_by_length[i].second;
		char *d_c = memo_mana_c.gpu_malloc(hashstrs_length[id]);
		//LOG(logger, "%d[old %d]th hashstrs: %s", i, id, hashstrs_buffer[id]); 
		safeCudaCall(cudaMemcpy(d_c, hashstrs_buffer[id], sizeof(char)*hashstrs_length[id], cudaMemcpyHostToDevice));
		h_hashstrs[i] = d_c;
		h_hashstrs_length[i] = hashstrs_length[id];
	}
	char **d_hashstrs = memo_mana_c.copy_bucks(h_hashstrs, doc_num);

	int *d_hashstrs_length = memo_mana_i.gpu_malloc(doc_num);
	safeCudaCall(cudaMemcpy(d_hashstrs_length, h_hashstrs_length, doc_num * sizeof(int), cudaMemcpyHostToDevice));

	for(int i=0, x=0, y=0;i<doc_num;i++) {
		int xl = order_by_length[i].first * THRESHOLD;
		int yl = order_by_length[i].first * (2.0-THRESHOLD);
		while(x<doc_num && order_by_length[x].first < xl) x++;
		while(y+1<doc_num && order_by_length[y+1].first <= yl) y++;
		startId[i] = x;
		endedId[i] = min(y, i);
		//startId[i] = 0;
		//endedId[i] = doc_num-1;
		LOG(logger, "Test range of doc_%06d is[%6d, %6d]", i, x, y);
	
	}

	int *d_startId = memo_mana_i.gpu_malloc(doc_num);
	safeCudaCall(cudaMemcpy(d_startId, startId, sizeof(int) * doc_num, cudaMemcpyHostToDevice));
	int *d_endedId = memo_mana_i.gpu_malloc(doc_num);
	safeCudaCall(cudaMemcpy(d_endedId, endedId, sizeof(int) * doc_num, cudaMemcpyHostToDevice));

	for(int i=0;i<doc_num;i++) {
		candies[i].clear();
	}
	LOG(logger, "Params: blocks = %d, threads = %d, method = %d", blocks, threads, method);
	if (this->blocks == -1) {
		LOG(logger, "%s", "No set blocks, threads and method, use default value!!!");
		this->method = 1;
	}

	if (this->method == 1) {
		LOG(logger, "%s", "use method 1");
		useMethod1(doc_num, d_hashstrs, d_hashstrs_length, d_startId, d_endedId);
	} else {
		LOG(logger, "%s", "use method 3");
		useMethod3(doc_num, d_hashstrs, d_hashstrs_length, d_startId, d_endedId);
	}

	core_time = clock() - ttt;
	LOG(logger, "hashstrs average length: %lf, max length: %lf", sumL / doc_num, maxL);
	LOG(logger, "content max length: %d", maxLD);
	LOG(logger, "calculateDups total time: %lf s", (clock()-ttt) / (double)CLOCKS_PER_SEC);
}

vector<int> DocDupDetectorGPU::get_candidate_dup_docs(int did) {
	return candies[did];
}

void DocDupDetectorGPU::refine(){
	for(int id1 = 0; id1<(int)hashstrs_buffer.size();id1++){
		const char* doc1=contents_buffer[id1];
		vector<int> item = candies[id1];
		for(int i=0;i<item.size();i++){
			const char *doc2=contents_buffer[item[i]];
			if(id1 < item[i] && score(doc1, doc2) >= MINIMUM_SCORE){
				real_dups[id1].push_back(item[i]);
				real_dups[item[i]].push_back(id1);
			}
		}
		LOG(logger, "refine doc_id = %d ended, candidates = %d, real_dups = %d", id1, (int)item.size(), real_dups[id1].size());
		if(item.size() && item.size() < 5) {
			for(int i=0;i<(int)item.size();i++) cout<<item[i]<<' '; cout<<endl;
		}
	}
}

vector<int> DocDupDetectorGPU::get_real_dup_docs(int did) {
	return real_dups[did];
}

DocDupDetectorGPU::~DocDupDetectorGPU() {
	for(int i=0;i<(int)contents_buffer.size();i++) {
		delete[] contents_buffer[i];
		delete[] hashstrs_buffer[i];
	}
}
