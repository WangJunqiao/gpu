#include "MatrixFileReader.h"

#include <stdio.h>
#include <memory.h>
#include <stdlib.h>

#include <vector>
#include <unordered_map>

#include "WordSimCalc.h"
using namespace std;

MatrixFileReader::MatrixFileReader(Logger *logger) {
	this->logger = logger;
	pos = vector<LL>();
}

vector<int> MatrixFileReader::init_reader(const char *matrix_file, const char *word_file) {
	LOG(logger, "%s", "Initialize reader...");
	
	FILE *fp = fopen(word_file, "r");
	int id, num;
	LL p;
	vector<int> ret;
	while(fscanf(fp, "%d", &id)!=EOF) {
		assert(id == pos.size());
		fscanf(fp, "%*s %I64d %d", &p, &num);
		ret.push_back(num);
		pos.push_back(p);
	}
	fclose(fp);
	mat_fp = fopen(matrix_file, "rb");
	if(logger)
		logger->printf("Initialize reader completely.");
	this->word_num = ret.size();
	this->r_iptr = (int**) malloc(word_num * sizeof(int*));
	this->r_fptr = (float**) malloc(word_num * sizeof(float*));
	for(int i=0;i<word_num;i++) {
		r_iptr[i] = NULL;
		r_fptr[i] = NULL;
	}
	return ret;
}

bool MatrixFileReader::load_data(int id) {
	if(id >= word_num) return false;

	if(r_iptr[id] != NULL)
		return true;
	
	fseek(mat_fp, pos[id], 0); /// change from _fseeki64(...)

	int num;
	fread(&num, INT_SIZE, 1, mat_fp);
	char *buffer = (char*)malloc((INT_SIZE+FLT_SIZE)*(num+1)); //freed
	memcpy(buffer, &num, INT_SIZE);
	fread(buffer+INT_SIZE, FLT_SIZE, 1, mat_fp);
	fread(buffer+INT_SIZE+FLT_SIZE, INT_SIZE+FLT_SIZE, num, mat_fp);
	
	r_iptr[id] = (int*)malloc(INT_SIZE * (num+1));
	r_fptr[id] = (float*)malloc(FLT_SIZE * (num+1));
	char *p = buffer;
	for(int i=0;i<=num;i++) {
		memcpy(r_iptr[id]+i, p, INT_SIZE);
		p += INT_SIZE;
		memcpy(r_fptr[id]+i, p, FLT_SIZE);
		p += FLT_SIZE;
	}
	free(buffer);
	return true;
}

bool MatrixFileReader::load_data(int start_id, int length) {
	for(int i=0;i<length;i++) {
		if(!load_data(start_id+i))
			return false;
	}
	return true;
}


void MatrixFileReader::destroy_reader() {
	for(int i=0;i<word_num;i++) {
		if(r_iptr[i]) free(r_iptr[i]);
		if(r_fptr[i]) free(r_fptr[i]);
	}
	free(this->r_iptr);
	this->r_iptr = NULL;
	
	free(this->r_fptr);
	this->r_fptr = NULL;

//	delete pos;
}

MatrixFileReader::~MatrixFileReader() {
	destroy_reader();
}
