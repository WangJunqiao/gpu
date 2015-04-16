#include "Common/Common.h"
#include "Demo.h"

#include <iostream>

#define N 4096
#define M 4096

#define BLOCKSIZE 16 //最多64
#define GRIDSIZE 64

#define LOAD_ROW BLOCKSIZE*GRIDSIZE   //每次load这么多到显存里面



float mi[N][M];
float si[N][N], fsi[N][N];
float mm[N * M], ss[N * N];

//类似矩阵乘法的方式
__global__ void calc_similarity(float *dev1, float *dev2, float *ans, int row) {
	// Block index
	int bx = blockIdx.x;
	int by = blockIdx.y;
	
	// Thread index
	int tx = threadIdx.x;
	int ty = threadIdx.y;

	//bx * tx = by * ty = LOAD_ROW

	int start1 = bx * BLOCKSIZE * M;
	int start2 = by * BLOCKSIZE * M;
	int base = tx * M + ty;

	float sum = 0.0f;
	for (int c = 0; c < M; c += BLOCKSIZE) {
		__shared__ float a[BLOCKSIZE][BLOCKSIZE+1];
		__shared__ float b[BLOCKSIZE][BLOCKSIZE+1];
		a[tx][ty] = dev1[start1 + base + c];
		b[tx][ty] = dev2[start2 + base + c];
		__syncthreads();
		for (int k = 0; k < BLOCKSIZE; ++k){
			sum += (a[tx][k] > 0 && b[ty][k] > 0) ? (a[tx][k] + b[ty][k]) : 0;
			//sum += a[tx][k] * b[ty][k];
		}
		__syncthreads();
	}
	int tot = gridDim.x * blockDim.x;
	ans[(bx * BLOCKSIZE + tx) * tot + by * BLOCKSIZE + ty] = sum;
}

void force() {
	for (int i = 0; i < N; i ++) {
		for (int j = 0; j < N; j ++) {
			float sum = 0.0;
			for (int k = 0; k < M; k ++) {
				sum += (mi[i][k] > 0 && mi[j][k] > 0) ? (mi[i][k] + mi[j][k]) : 0;
				//sum += mi[i][k] * mi[j][k];
			}
			fsi[i][j] = sum;
		}
	}
}

int test_word_sim() {
	for (int i = 0; i < N; i ++) {
		for (int j = 0; j < M; j ++) {
			mi[i][j] = (rand() % 4 == 0) ? 0.0f : (rand() % 100 / 100.0f); //随机生成数据
			mm[i * M + j] = mi[i][j]; //赋值给一维数组
		}
	}
	float *dev1, *dev2, *ans;
	safeCudaCall(cudaMalloc(&dev1, sizeof(float) * LOAD_ROW * M));
	safeCudaCall(cudaMalloc(&dev2, sizeof(float) * LOAD_ROW * M));
	safeCudaCall(cudaMalloc(&ans, sizeof(float) * LOAD_ROW * LOAD_ROW));
	printf("begin gpu computing...\n");
	clock_t cpu = 0, gpu = 0, cur = clock();
	for (int row1 = 0; row1 < N; row1 += LOAD_ROW) {
		safeCudaCall(cudaMemcpy(dev1, mm + row1 * M, sizeof(float) * LOAD_ROW * M, cudaMemcpyHostToDevice));
		for (int row2 = 0; row2 < N; row2 += LOAD_ROW) {
			safeCudaCall(cudaMemcpy(dev2, mm + row2 * M, sizeof(float) * LOAD_ROW * M, cudaMemcpyHostToDevice));
			// Setup execution parameters
			dim3 threads(BLOCKSIZE, BLOCKSIZE);
			dim3 grid(GRIDSIZE, GRIDSIZE);
			calc_similarity<<<grid, threads>>>(dev1, dev2, ans, LOAD_ROW);
			cudaDeviceSynchronize();

			safeCudaCall(cudaMemcpy(ss, ans, sizeof(float) * LOAD_ROW * LOAD_ROW, cudaMemcpyDeviceToHost));
			for (int i = 0; i < LOAD_ROW; i ++) {
				for (int j = 0; j < LOAD_ROW; j ++) {
					si[row1 + i][row2 + j] = ss[i * LOAD_ROW + j];
				}
			}
		}
		printf("row1 = %d ended\n", row1);
	}
	cudaFree(dev1);
	cudaFree(dev2);
	cudaFree(ans);
	gpu += clock() - cur;
	printf("gpu = %lf s\n", gpu / (double)CLOCKS_PER_SEC);

	cur = clock();
	force();
	cpu += clock() - cur;

	printf("cpu = %lf s, gpu = %lf s, speed up = %lf\n", cpu / (double)CLOCKS_PER_SEC, gpu / (double)CLOCKS_PER_SEC, cpu / (double)gpu);

	float ma = 0.0;
	for (int i = 0; i < N; i ++) {
		for (int j = 0; j < N; j ++) {
			ma = max(ma, fabs(si[i][j] - fsi[i][j]));
		}
	}
	printf("max diff = %f\n", ma);

	int x, y;
	while (cin >> x >> y) {
		printf("gpu value = %f, cpu value = %f\n", si[x][y], fsi[x][y]);
	}
    return 0;
}

