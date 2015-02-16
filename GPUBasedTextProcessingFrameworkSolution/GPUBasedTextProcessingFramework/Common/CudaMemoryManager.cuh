#ifndef CUDA_MEMORY_MANAGER_H
#define CUDA_MEMORY_MANAGER_H

#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "Common.h"

using namespace std;
/*

*/
template<typename T> class CudaMemoryManager{
public:
	CudaMemoryManager() {
		v1.clear();
		v2.clear();
	}
	
	///allocate length * sizeof(T) bytes gpu memory, return head address.
	T* gpu_malloc(int length) {
		//printf("CudaMemoryManager: malloc %d\n", length);
		T *d_addr;
		safeCudaCall(cudaMalloc(&d_addr, sizeof(T) * length));
		v1.push_back(d_addr);
		return d_addr;
	}

	T* gpu_malloc_and_copy(int length, T* h_addr) {
		T* d_addr = gpu_malloc(length);
		safeCudaCall(cudaMemcpy(d_addr, h_addr, sizeof(T)*length, cudaMemcpyHostToDevice));
		return d_addr;
	}

	///将一个指针数组拷贝到GPU中，并且返回首地址	
	T** copy_bucks(T** h_addrs, int length) { 
		T** d_addrs;
		safeCudaCall(cudaMalloc(&d_addrs, sizeof(T*) * length));
		safeCudaCall(cudaMemcpy(d_addrs, h_addrs, length * sizeof(T*), cudaMemcpyHostToDevice));
		v2.push_back(d_addrs);
		return d_addrs;
	}
	
	///deprecated
	///申请一个GPU内部的二维数组，大小row*col，返回首地址，并赋值原始一维数组地址
	T** gpu_malloc2D(int row, int col, T* &out_raw_addr) { //申请一个row*col的二维数组，并且返回首地址
		T** h_addrs = (T**) malloc(sizeof(T*) * row);
		out_raw_addr = gpu_malloc(row * col);
		for(int i=0;i<row;i++) {
			h_addrs[i] = out_raw_addr + i * col;
		}
		T** d_addrs = copy_bucks(h_addrs, row);
		v2.push_back(d_addrs);
		free(h_addrs);
		return d_addrs;
	}

	void gpu_free(T* d_arr) {
		for(int i=0;i<(int)v1.size();i++) {
			if(v1[i] == d_arr) {
				safeCudaCall(cudaFree(d_arr));
				v1.erase(v1.begin() + i);
				return;
			}
		}
	}

	void gpu_free2D(T** d_arr2) {
		for(int i=0;i<(int)v2.size();i++) {
			if(v2[i] == d_arr2) {
				safeCudaCall(cudaFree(d_arr2));
				v2.erase(v2.begin() + i);
				return;
			}
		}
	}

	void free_all() {
		printf("Freeing cuda memory, %d of T*, %d of T**, ", (int)v1.size(), (int)v2.size());
		while(!v1.empty()) {
			safeCudaCall(cudaFree(v1.back()));
			v1.pop_back();
		}
		while(!v2.empty()) {
			safeCudaCall(cudaFree(v2.back()));
			v2.pop_back();
		}
		printf("complete successfully.\n");
	}

	~CudaMemoryManager() {
		free_all();
	}

private:
	vector<T*> v1;
	vector<T**> v2;
	//DISALLOW_COPY_AND_ASSIGN(CudaMemoryManager);
};


#endif