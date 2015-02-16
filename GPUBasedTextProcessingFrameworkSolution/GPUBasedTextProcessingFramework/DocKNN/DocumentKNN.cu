#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "thrust/device_vector.h"
#include "thrust/host_vector.h"
#include "thrust/sort.h"

#include "../Common/Common.h"
#include "DocumentKNN.h"

__global__ void MatrixMlutKernel(DataType *query_data, DataType *data, int num, int query_num, int dim, DataType *result) {
	int row = blockIdx.x*blockDim.x + threadIdx.x;
	int col = blockIdx.y*blockDim.y + threadIdx.y;
	if(row < query_num && col < num) {
		//printf("row:%d, col:%d\n", row, col);
		DataType xx  = 0, yy = 0, xy = 0;
		for(int i = 0; i < dim; i++) {
			xx += query_data[row*dim + i]*query_data[row*dim + i];
			yy += data[col*dim + i]*data[col*dim + i];
			xy += data[col*dim + i]*query_data[row*dim + i];
		}
		if(xx == 0 || yy == 0)result[row*num + col] = 1;
		else result[row*num + col] = 1 - xy/(sqrt(xx)*sqrt(yy));
	}
}

void DOCKNN::calc_gpu_knn(){
	int num = m.get_num(), query_num = m.get_query_num(), dim = m.get_dim(); 

	safeCudaCall(cudaSetDevice(0));
	DataType *d_data, *d_query_data, *d_result, *h_result;
	h_result = (DataType*)malloc(num*query_num*sizeof(DataType));
	safeCudaCall(cudaMalloc(&d_data, num*dim*sizeof(DataType)));	
	safeCudaCall(cudaMalloc(&d_query_data, query_num*dim*sizeof(DataType)));
	safeCudaCall(cudaMalloc(&d_result, num*query_num*sizeof(DataType)));
	
	safeCudaCall(cudaMemcpy(d_data, this->m.get_data(), num*dim*sizeof(DataType), cudaMemcpyHostToDevice));
	safeCudaCall(cudaMemcpy(d_query_data, this->m.get_query_data(), query_num*dim*sizeof(DataType), cudaMemcpyHostToDevice));

	dim3 dimMultBlock(BLOCK_SIZE, BLOCK_SIZE);
	dim3 dimMultGrid(query_num/dimMultBlock.x+1, num/dimMultBlock.y+1);
	time_t t = clock();
	LOG(logger, "started kernel func.");
	MatrixMlutKernel<<<dimMultGrid, dimMultBlock>>>(d_query_data, d_data, num, query_num, dim, d_result);
	safeCudaCall(cudaDeviceSynchronize());
	safeCudaCall(cudaMemcpy(h_result, d_result, num*query_num*sizeof(DataType), cudaMemcpyDeviceToHost));
	LOG(logger, "finished kernel func.");
	
	vector<pair<DataType, int> > h_vec;
	for(int i = 0; i < num*query_num; i++)
		h_vec.push_back(make_pair(h_result[i], (i+1)%num));
	for(int i = 0; i < query_num; i++) {
		sort(h_vec.begin()+i*num, h_vec.begin()+(i+1)*num);
	}
	for(int i = 0; i < query_num; i++) {
		for(int j = 0; j < this->k; j++) {
			this->gpu_result.push_back(make_pair(h_vec[i*num+j].second, h_vec[i*num+j].first));
		}
	}
	time_t t2 = clock();
	LOG(logger, "gpu time: %d ms.", t2-t);
}

void DOCKNN::calc_cpu_knn(){
	int num = m.get_num(), query_num = m.get_query_num(), dim = m.get_dim();
	pair<DataType, int> **result = new pair<DataType, int> *[query_num];
	for(int i = 0; i < query_num; i++)
		result[i] = new pair<DataType, int>[num];
	time_t t1 = clock();

	LOG(logger, "started calculation on cpu.");
	DataType *data = m.get_data(), *query_data = m.get_query_data();

	for(int i = 0; i < query_num; i++) {
		for(int j = 0; j < num; j++) {
			DataType xx = 0, yy = 0, xy = 0;
			for(int k = 0; k < dim; k++) {
				xx += query_data[i*dim+k]*query_data[i*dim+k];
				yy += data[j*dim+k]*data[j*dim+k];
				xy += data[j*dim+k]*query_data[i*dim+k];
			}
			DataType r;
			if(xx == 0 || yy == 0)r = 1;
			else r = 1 - xy/(sqrt(xx)*sqrt(yy));
			result[i][j] = make_pair(r, j+1);
		}
	}
	LOG(logger, "finished calculation on cpu.");
	for(int i = 0; i < query_num; i++)
		sort(result[i], result[i]+num);
	for(int i = 0; i < query_num; i++) {
		for(int j = 0; j < k; j++) {
			cpu_result.push_back(make_pair(result[i][j].second, result[i][j].first));
		}
	}
	for (int i = 0; i < query_num; i++) {
		delete[] result[i];
		result[i] = NULL;
	}
	delete[] result;
	result = NULL;
	time_t t2 = clock();
	LOG(logger, "cpu time: %d ms.", t2-t1);
}

void DOCKNN::print_gpu_result() {
	int query_num = m.get_query_num();
	printf("the query results of gpu are:\n");
	for(int i = 0; i < query_num; i++) {
		printf("Doc %d:", i+1);
		for(int j = 0; j < k; j++)
			printf("(%d, %.3lf)%c", gpu_result[i*k+j].first, gpu_result[i*k+j].second, j+1==k?'\n':' ');
	}
	printf("finished.\n");
}

void DOCKNN::print_cpu_result() {
	int query_num = m.get_query_num();
	printf("the query results of cpu are:\n");
	for(int i = 0; i < query_num; i++) {
		printf("Doc %d:", i+1);
		for(int j = 0; j < k; j++)
			printf("(%d, %.3lf)%c", cpu_result[i*k+j].first, cpu_result[i*k+j].second, j+1==k?'\n':' ');
	}
	printf("finished.\n");
}

void DOCKNN::save_gpu_result(char *path) {
	FILE *fp = fopen(path, "w");
	int query_num = m.get_query_num();
	fprintf(fp, "the query results of gpu are (word, distance):\n");
	for(int i = 0; i < query_num; i++) {
		fprintf(fp, "Doc %d:", i+1);
		for(int j = 0; j < k; j++)
			fprintf(fp, "(%d, %.3lf)%c", cpu_result[i*k+j].first, cpu_result[i*k+j].second, j+1==k?'\n':' ');
	}
	fclose(fp);
	LOG(logger, "the gpu result has saved in file.");
}

void DOCKNN::save_cpu_result(char *path) {
	FILE *fp = fopen(path, "w");
	int query_num = m.get_query_num();
	fprintf(fp, "the query results of cpu are (word, distance):\n");
	for(int i = 0; i < query_num; i++) {
		fprintf(fp, "Doc %d:", i+1);
		for(int j = 0; j < k; j++)
			fprintf(fp, "(%d, %.3lf)%c", cpu_result[i*k+j].first, cpu_result[i*k+j].second, j+1==k?'\n':' ');
	}
	fclose(fp);
	LOG(logger, "the cpu result has saved in file.");
}