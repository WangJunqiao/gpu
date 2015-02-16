#include "WordSimCalcGPUImpl.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "MatrixFileReader.h"
#include "../Common/Logger.h"
#include "../Common/CudaMemoryManager.cuh"

using namespace std;


GPUWordSimCalculator::GPUWordSimCalculator(Logger *logger, const string &result_dir, int top_words_num) 
	: WordSimCalculator(logger, result_dir, top_words_num){
		block_num = 128;
		thread_num = 128;
		pairs_limit = 30000000;
		setted = 0;
}

void GPUWordSimCalculator::set_params(int block_num, int thread_num, int pairs_limit) {
	this->block_num = block_num;
	this->thread_num = thread_num;
	this->pairs_limit = pairs_limit;
	this->setted = 1; //set flag
}

__global__ void calc_parallel(PIF **data1, PIF **data2, PII *dev_jobs, float *dev_ans, int job_num) {
	int tot_threads = gridDim.x * blockDim.x;
	int thread_id = threadIdx.x + blockDim.x * blockIdx.x;

	for(int jid=thread_id;jid<job_num;jid+=tot_threads) {
		int id1 = dev_jobs[jid].first;
		int id2 = dev_jobs[jid].second;
		int v_size1 = data1[id1][0].first;
		int v_size2 = data2[id2][0].first;

		float sum1 = 0.0f;
		PIF *p1 = data1[id1];
		PIF *p2 = data2[id2];

		for(int i=1, j=1, iv1 = p1[1].first, iv2 = p2[1].first;i<=v_size1 && j<=v_size2;){
			if(iv1 < iv2) {
				i++;
				if(i>v_size1) break;
				iv1 = p1[i].first;
				continue;
			}
			if(iv1 > iv2) {
				j++;
				if(j>v_size2) break;
				iv2 = p2[j].first;
				continue;
			}
			sum1 += (p1[i].second+p2[j].second);
			//sum1 += i;
			i++;j++;
			if(i>v_size1 || j>v_size2) break;
			iv1 = p1[i].first;
			iv2 = p2[j].first;
		}
		dev_ans[jid] = sum1 / (data1[id1][0].second+data2[id2][0].second); 
	}
}

//PIF pif[TOP_WORDS_NUM];
//PIF* ppif[TOP_WORDS_NUM];

vector<PIF> pif;
vector<PIF*> ppif;

//将iptr_arr和fptr_arr数组所指向的内容拷贝到显存，并且在内存和显存中都开辟了一个指针数组
PIF** array_copy_host_to_device(int** ip, float** fp, int num) {
	for(int i=0;i<num;i++) {
		int v_size = ip[i][0];
		pif.clear();
		for(int j=0;j<=v_size;j++) {
			pif.push_back(make_pair(ip[i][j], fp[i][j]));
		}
		ppif.push_back(NULL);
		safeCudaCall(cudaMalloc(&ppif[i], sizeof(PIF)*(v_size+1)));
		safeCudaCall(cudaMemcpy(ppif[i], pif.data(), sizeof(PIF)*(v_size+1), cudaMemcpyHostToDevice));
	}
	PIF** dev_add;

	safeCudaCall(cudaMalloc(&dev_add, sizeof(PIF*)*num));
	safeCudaCall(cudaMemcpy(dev_add, ppif.data(), sizeof(PIF*)*num, cudaMemcpyHostToDevice));

	return dev_add;
}

//PIF* _ppif[TOP_WORDS_NUM];
void free_cuda_memory(PIF** dev_add, int num) {
	vector<PIF*> _ppif(num, NULL);
	cudaMemcpy(_ppif.data(), dev_add, sizeof(PIF*)*num, cudaMemcpyDeviceToHost);
	for(int i=0;i<num;i++) {
		safeCudaCall(cudaFree(_ppif[i]));
	}
	cudaFree(dev_add);
}

vector<PII> get_best_sequence(int blocks) {
	// 	int x = 0, y = blocks-1;
	// 	vector<PII> seq;
	// 	for(int r=0;r<blocks;r++) {
	// 		if(r%2==0) {
	// 			for(int i=x;i<=y;i++) {
	// 				seq.push_back(PII(x, i));
	// 			}
	// 			x++;
	// 		} else {
	// 			for(int i=y;i>=x;i--) {
	// 				seq.push_back(PII(y, i));
	// 			}
	// 			y--;
	// 		}
	// 	}
	// 	return seq;
	vector<PII> seq;
	for(int i=0;i<blocks;i++) for(int j=i;j<blocks;j++) seq.push_back(PII(i, j));
	return seq;
}

/*
According to the first order word similarity matrix, iteratively calculate the next order word similarity matrix.
Word similarity matrix is a symmetrical float value matrix.
The number of words may be very large(e.g. 200000), and because of the computation mode of first order matrix, the word similarity matrix is not full which means it has lots of zero value.
 
*/
void GPUWordSimCalculator::calc_similarity_matrix() {
	clock_t ttt = clock();

	LOG(logger, "---------Begin Calculate Second Order Sim Matrix---------");
	if(setted) {
		LOG(logger, "Use user settings: block number = %d, thread number = %d, pair_limit = %d", 
			block_num, thread_num, pairs_limit);
	} else {
		LOG(logger, "Warnning! Use default settings: block number = %d, thread number = %d, pair_limit = %d", 
			block_num, thread_num, pairs_limit);
	}
	Logger reader_logger(stdout);
	MatrixFileReader reader(&reader_logger);

	vector<int> nums = reader.init_reader(
		get_matrix_file_name(1).c_str(), 
		get_word_file_name().c_str()
		);
	LOG(logger, "init reader, tot word number %d", (int)nums.size());

	vector<int> block_start;

	vector<int> belong_to; //
	for(int i=0;i<(int)nums.size();) {
		int cur = 0, j = i;
		while(j<(int)nums.size() && cur + nums[j] <= this->pairs_limit) {
			cur += nums[j];
			j++;
		}
		for(int k=i;k<j;k++) belong_to.push_back((int)block_start.size());
		block_start.push_back(i);
		i = j;
	}
	block_start.push_back((int)nums.size());

	int tot_block = (int)block_start.size()-1;
	LOG(logger, "tot_block = %d\n", tot_block);

	for(int i=0;i<(int)block_start.size()-1;i++) {
		LOG(logger, "block %d: [%d, %d)", i, block_start[i], block_start[i+1]);
	}

	clock_t tt = clock(), t;
	clock_t gpu_time, write_time, data_copy_time, data_release_time, read_time;
	gpu_time = write_time = data_copy_time = data_release_time = read_time = 0;

	vector<PII> seq = get_best_sequence(tot_block);

	FILE *fp = fopen(get_matrix_file_name(2).c_str(), "wb");
	long long all_pairs = 0;

	vector<vector<PII> >pairs(this->top_words_num, vector<PII>());

	for(int p=0;p<(int)seq.size();) {
		int b1 = seq[p].first;
		int start1 = block_start[b1], end1 = block_start[b1+1], len1 = end1 - start1;

		t = clock();
		reader.load_data(start1, len1);
		read_time += clock() - t;

		t = clock();
		for(int i=0;i<tot_block;i++) pairs[i].clear();
		for(int i=0;i<len1;i++) { //处理所有需要计算的pairs，并且把他们放到各自的队列中
			for(int j=1;j<=reader.r_iptr[i+start1][0];j++) {
				int k = reader.r_iptr[i+start1][j];
// 				if(k<0 || k >= this->top_words_num) {
// 					printf("k = %d\n", k);
// 				}
// 				if(belong_to[k]<0 || belong_to[k]>=tot_block) {
// 					printf("k = %d, belong[k] = %d\n", k, belong_to[k]);
// 				}
				if(belong_to[i+start1]==belong_to[k] && i+start1 > k) 
					continue; //同一块内部，不要重复计算

				pairs[belong_to[k]].push_back(PII(i+start1, k));

				//}
			}
		}
		LOG(logger, "Prepare all pairs need to be calc, time used = %d ms", clock()-t);

		t = clock();
		PIF** dev_data1 = array_copy_host_to_device(reader.r_iptr+start1, reader.r_fptr+start1, end1-start1);
		data_copy_time += clock()-t;

		int sp = p;
		for(;p<(int)seq.size() && seq[p].first==seq[sp].first;p++) {
			int b2 = seq[p].second, pb2_sz = (int)pairs[b2].size();

			if(pb2_sz == 0) {
				LOG(logger, "0 pairs need to be calc, skip this block!");
				continue;
			}
			LOG(logger, "Ready to calc %d pairs with block(%d, %d), %d/%d completed..", 
					pb2_sz, b1, b2, p, (int)seq.size());

			int start2 = block_start[b2], end2 = block_start[b2+1], len2 = end2 - start2;

			t = clock();
			reader.load_data(start2, len2);
			read_time += clock() - t;

			t = clock();
			PIF** dev_data2 = array_copy_host_to_device(reader.r_iptr+start2, reader.r_fptr+start2, len2);
			data_copy_time += clock()-t;

			t = clock();
			int job_num = (int)pairs[b2].size();
			vector<PII> jobs(job_num, PII());
			vector<float> ans(job_num, 0.0f);
			for(int x=0;x<job_num;x++) {
				jobs[x] = pairs[b2][x];
				jobs[x].first -= block_start[belong_to[jobs[x].first]];
				jobs[x].second -= block_start[belong_to[jobs[x].second]];
			}

			PII *dev_jobs;
			safeCudaCall(cudaMalloc(&dev_jobs, sizeof(PII)*job_num)); //freed
			safeCudaCall(cudaMemcpy(dev_jobs, jobs.data(), sizeof(PII)*job_num, cudaMemcpyHostToDevice));

			float *d_ans;
			safeCudaCall(cudaMalloc(&d_ans, FLT_SIZE * pairs[b2].size()));//freed
			data_copy_time += clock()-t;

			t = clock();
			calc_parallel<<<block_num, thread_num>>>(dev_data1, dev_data2, dev_jobs, d_ans, job_num);
			cudaDeviceSynchronize();
			gpu_time += (clock()-t);
			LOG(logger, "GPU calculate, time used=%d ms", (int)(clock()-t));

			t = clock();
			safeCudaCall(cudaMemcpy(ans.data(), d_ans, FLT_SIZE*job_num, cudaMemcpyDeviceToHost));
			data_copy_time += clock()-t;

			all_pairs += pairs[b2].size();

			t = clock();
			for(int i=0;i<(int)pairs[b2].size();i++) {
				pair<int, int> p = pairs[b2][i];
				fwrite(&p.first, sizeof(int), 1, fp);
				fwrite(&p.second, sizeof(int), 1, fp);
				fwrite(&ans[i], sizeof(float), 1, fp);
			}
			write_time += (clock()-t);
			LOG(logger, "dispatch %d pairs' result, time used=%d ms\n", 
					(int)pairs[b2].size(), clock()-t);

			t = clock();
			free_cuda_memory(dev_data2, len2);
			cudaFree(dev_jobs);
			cudaFree(d_ans);
			data_release_time += clock()-t;
			puts("");
			fflush(stdout);
		}
		t = clock();
		free_cuda_memory(dev_data1, len1);
		data_release_time += clock()-t;

		printf("b1 = %d, ended.\n", b1);
	}

	printf("all calc pairs = %lld\n", all_pairs);
	printf("reader reading_time: %d ms\n", read_time);
	printf("GPU time: %d ms\n", gpu_time);
	printf("write time: %d ms\n", write_time);
	printf("data copy time: %d ms\n", data_copy_time);
	printf("data release time: %d ms\n", data_release_time);
	printf("tot time: %d ms\n", (int)(clock()-tt));
	fclose(fp);

	core_time = clock() - ttt;

	rebuild_triples(2, 2);

	
}