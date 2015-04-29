#ifndef COMMON_H
#define COMMON_H
#include <stdio.h>
#include <string>
#include <time.h>
#include <stdarg.h>

using namespace std;

#define DEBUG

#ifdef DEBUG
    #define LOG(logger, format, ...) if(logger)(logger)->printf(format, __VA_ARGS__)
#else
	#define LOG(logger, format, ...) 
#endif


//disable class's copy and assignment operator, used in the private: area.
#define DISALLOW_COPY_AND_ASSIGN(TypeName) \
	TypeName(const TypeName&);             \
	void operator=(const TypeName&)

#define safeCudaCall(func)  { \
	cudaError_t error_code; \
	if((error_code = func) != cudaSuccess) {\
	printf("error occurs, error code: %d, error string: %s, %s(line: %d)\n", (int)error_code, cudaGetErrorString(error_code), __FILE__, __LINE__); \
	exit(-1); \
	} }

typedef long long LL;
typedef pair<int, int> PII;
typedef pair<int, float> PIF;

const int INT_SIZE = sizeof(int);
const int FLT_SIZE = sizeof(float);

#endif
