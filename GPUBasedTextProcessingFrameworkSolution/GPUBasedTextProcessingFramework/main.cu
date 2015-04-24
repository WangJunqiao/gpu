#include "Demo.h"

#include <stdio.h>
#include <string.h>
#include <cuda_runtime.h>

bool initCUDA() {
	int count = 0;
	int i = 0;
	cudaGetDeviceCount(&count); //看看有多少个设备?
	if(count == 0) {  //哈哈~~没有设备.
		fprintf(stderr, "There is no device.\n");
		return false;
	}
	cudaDeviceProp prop;
	for(i = 0; i < count; i++)  {//逐个列出设备属性:
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
// 			if(prop.major >= 1) {
// 				break;
// 			}
			cudaDeviceProp sDevProp = prop;
			printf( "%d \n", i);
			printf( "Device name: %s\n", sDevProp.name );
			printf( "Device memory: %lld\n", (long long)sDevProp.totalGlobalMem );
			printf( "Shared Memory per-block: %d\n", (int)sDevProp.sharedMemPerBlock );
			printf( "Register per-block: %d\n", sDevProp.regsPerBlock );
			printf( "Warp size: %d\n", sDevProp.warpSize );
			printf( "Memory pitch: %lld\n", (long long)sDevProp.memPitch );
			printf( "Constant Memory: %lld\n", (long long)sDevProp.totalConstMem );
			printf( "Max thread per-block: %d\n", sDevProp.maxThreadsPerBlock );
			printf( "Max thread dim: ( %d, %d, %d )\n", sDevProp.maxThreadsDim[0],
				sDevProp.maxThreadsDim[1], sDevProp.maxThreadsDim[2] );
			printf( "Max grid size: ( %d, %d, %d )\n", sDevProp.maxGridSize[0],  
				sDevProp.maxGridSize[1], sDevProp.maxGridSize[2] );
			printf( "Compute ability Ver: %d.%d\n", sDevProp.major, sDevProp.minor );
			printf( "Clock: %d\n", sDevProp.clockRate );
			printf( "textureAlignment: %d\n", sDevProp.textureAlignment );
			cudaSetDevice(i);
			printf("\n CUDA initialized.\n");
		}
	}
// 	if(i == count) {
// 		fprintf(stderr, "There is no device supporting CUDA.\n");
// 		return false;
// 	}

	return true;
}

void print_usage_main() {
	puts("Demo Usage   -options");
	puts("-word_sim           word similarity calculation");
	puts("-doc_dup            document duplicate detection");
	puts("-doc_clustering     document clustering");
}



int main(int argc, char** argv) {
	if (argc < 2) {
		print_usage_main();
		return 0;
	}
	initCUDA();
	if (strcmp(argv[1], "-word_sim") == 0) {
		word_similarity_test(argc, argv);
	} else if (strcmp(argv[1], "-doc_dup") == 0) {
		doc_dup_detection_test(argc, argv);
	} else if (strcmp(argv[1], "-doc_clustering") == 0) {
		doc_clustering_test(argc, argv);
	} else {
		print_usage_main();
		return 0;
	}
}