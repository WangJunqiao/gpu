#ifndef WORD_SIMILARITY_TEST_H
#define WORD_SIMILARITY_TEST_H

#include <stdio.h>
#include <utility>
#include <iostream>
#include <list>
#include <queue>
#include <vector>

#include "DataReader.h"
#include "WordSimCalc.h"
#include "WordSimCalcCPUImpl.h"
#include "WordSimCalcGPUImpl.h"
#include "WikipediaDataSource.h"
#include "../Common/Logger.h"

using namespace std;

const string matrix_file1 = "../data/matrix_file1";
const string matrix_file2 = "../data/matrix_file2";
const string word_file = "../data/word_file";


DataReader reader(NULL);

float get_sim(int id1, int id2) {
	int *i1, *i2;
	float *f1, *f2;
	reader.load_data(id1);
	i1 = reader.r_iptr[id1];
	f1 = reader.r_fptr[id1];

	reader.load_data(id2);
	i2 = reader.r_iptr[id2];
	f2 = reader.r_fptr[id2];

	float sum = 0;
	for(int i=1, j=1;i<=i1[0] && j<=i2[0];) {
		if(i1[i] < i2[j]) {
			i++;
		} else if(i1[i] > i2[j]) {
			j++;
		} else {
			sum += f1[i];
			sum += f2[j];
			i++;
			j++;
		}
	}
	return sum/(f1[0]+f2[0]);
}

void check() {
	reader.init_reader(matrix_file1.c_str(), word_file.c_str());

	DataReader reader2(NULL);
	reader2.init_reader(matrix_file2.c_str(), word_file.c_str());

	for(int i=0;reader2.load_data(i);i++) {
		int id1 = i, id2;
		int *id = reader2.r_iptr[id1];
		float *val = reader2.r_fptr[id1];
		//printf("%d\n", id[0]);
		for(int j=1;j<=id[0];j++) {
			id2 = id[j];
			float v = val[j];
			float v2;
			if(fabs(v - (v2 = get_sim(id1, id2))) > 1e-4) {
				printf("err!!!, id1 = %d, id2 = %d, v = %f, v2 = %f\n", id1, id2, v, v2);
				goto end;
			} else {
			//	printf("%d-%d right\n", id1, id2);
			}
		}
		if(i%1000 == 0) {
			printf("check %d successfully\n", i);
		}
	}
	
end:;
	printf("checked\n");
}



int test() {
	//Logger logger("../data/log_file.txt", false);
	Logger logger(stdout);
	WordSimCalculator *ins = new GPUWordSimCalculator(&logger, "../data/", 19809);
	ins->calc_first_order(new WikipediaDataSource());
	ins->calc_next_order_from(1);
	check();
	return 0;
}


#endif