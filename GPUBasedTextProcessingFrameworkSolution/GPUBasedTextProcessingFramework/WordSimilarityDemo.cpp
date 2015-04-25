#include "Demo.h"

#include <stdio.h>
#include <utility>
#include <iostream>
#include <list>
#include <queue>
#include <string.h>
#include <memory.h>
#include <math.h>
#include <vector>

#include <sys/types.h>
#include <dirent.h> 
#include <unistd.h>

#include "./WordSimilarity/MatrixFileReader.h"
#include "./WordSimilarity/WordSimCalc.h"
#include "./WordSimilarity/WordSimCalcCPUImpl.h"
#include "./WordSimilarity/WordSimCalcGPUImpl.h"
#include "./DataSource/WikipediaDataSource.h"
#include "./Common/Logger.h"

using namespace std;

const string matrix_file1 = "./data/matrix_file1";
const string matrix_file2 = "./data/matrix_file2";
const string word_file = "./data/word_file";


MatrixFileReader reader(NULL);

static float get_sim(int id1, int id2) {
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

static void check() {
	reader.init_reader(matrix_file1.c_str(), word_file.c_str());

	MatrixFileReader reader2(NULL);
	reader2.init_reader(matrix_file2.c_str(), word_file.c_str());

    int threshold = 1000;
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
		if(i == threshold) {
			printf("check %d successfully\n", i);
            threshold *= 2;
		}
	}

end:;
	printf("checked\n");
}

static void print_usage() {
	printf("Word Similarity Calculation Usage\n");
	printf("-cpu         test in CPU\n");
	printf("-gpu         test in GPU\n");
	printf("-corpus      big text corpus, e.g. ./data/wiki_plain_txt\n");
	printf("-top_words   the number of words that appear most times\n");
	printf("-win_size    the scan window size, we consider [-size, size]\n");
	printf("-no_cal_mi   No need to calculate mutual information matrix\n");
	printf("-log         log file name, e.g. ./log.txt\n");
	printf("-root_dir    where the word_file and mi_matrix is");
	printf("-output_dir  output directory, e.g. D:/aaa\n");
	printf("-max_docs    max documents, eg 6000000\n");
	printf("-set_param   set the blocks, threads and pair_limit\n");
	exit(0);
}

int word_similarity_test(int argc, char **argv) {
	Logger *file_logger = NULL;
	bool no_cal_mi = false;
	int win_size = 3;
	int top_words = 200000;
	WikipediaDataSource *wiki_src = NULL;
	int mask = 0;
	string root_dir = ".", output_dir = ".";
	int max_doc = 6000000; 
	int blocks = -1, threads = -1, pair_limits = -1;
	for (int i = 2; i < argc;) {
		if (strcmp(argv[i], "-log") == 0 && i + 1 < argc) {
			file_logger = new Logger(argv[i + 1], false);
            i += 2;
		} else if (strcmp(argv[i], "-no_cal_mi") == 0) {
			no_cal_mi = true;
            i ++;
		} else if (strcmp(argv[i], "-win_size") == 0 && i + 1 < argc) {
			win_size = atoi(argv[i + 1]);
            i += 2;
		} else if (strcmp(argv[i], "-top_words") == 0 && i + 1 < argc) {
			top_words = atoi(argv[i + 1]);
            i += 2;
		} else if (strcmp(argv[i], "-corpus") == 0 && i + 1 < argc) {
			wiki_src = new WikipediaDataSource(argv[i + 1]);
            i += 2;
		} else if (strcmp(argv[i], "-cpu") == 0) {
			mask |= 1;
            i ++;
		} else if (strcmp(argv[i], "-gpu") == 0) {
			mask |= 2;
            i ++;
		} else if (strcmp(argv[i], "-output_dir") == 0 && i + 1 < argc) {
			output_dir = argv[i + 1];
            i += 2;
		} else if (strcmp(argv[i], "-root_dir") == 0 && i + 1 < argc) {
			root_dir = argv[i + 1];
			i += 2;
		} else if (strcmp(argv[i], "-max_doc") == 0 && i + 1 < argc) {
			max_doc = atoi(argv[i + 1]);
            i += 2;
		} else if (strcmp(argv[i], "-set_param") == 0 && i + 3 < argc) {
			blocks = atoi(argv[i + 1]);
			threads = atoi(argv[i + 2]);
			pair_limits = atoi(argv[i + 3]);
			i += 4;
		} else {
			print_usage();
		}
	}
	assert(root_dir.back() != '/' && root_dir.back() != '\\');
	assert(output_dir.back() != '/' && output_dir.back() != '\\');

	Logger *logger = new Logger(stdout, file_logger);

    LOG(logger, "mask = %d", mask);

	clock_t cpu_core_time = 1, gpu_core_time = 1;
	
    system((string("mkdir -p ") + output_dir + "/cpu/").c_str());
    system((string("mkdir -p ") + output_dir + "/gpu/").c_str());
	if (mask & 1) {
		CPUWordSimCalculator *ins = new CPUWordSimCalculator(logger, root_dir, output_dir + "/cpu/", top_words, win_size);
		if (!no_cal_mi) {
			if (wiki_src == NULL) {
				print_usage();
			}
			wiki_src->set_max_docs(max_doc);
			ins->calc_mutual_info_matrix(wiki_src);
		} else {
			LOG(logger,"%s", "no calc mi!!");
		}
		ins->calc_similarity_matrix();
		//check();
		cpu_core_time = ins->core_time; 
		delete ins;
	}
	if (mask & 2) {
		GPUWordSimCalculator *ins = new GPUWordSimCalculator(logger, root_dir, output_dir + "/gpu/", top_words, win_size);
		if (blocks > 0) {
			ins->set_params(blocks, threads, pair_limits);
		}
		if (!no_cal_mi) {
			if (wiki_src == NULL) {
				print_usage();
			}
			wiki_src->set_max_docs(max_doc);
			ins->calc_mutual_info_matrix(wiki_src);
		} else {
			LOG(logger, "%s", "no calc mi!!!");
		}
		ins->calc_similarity_matrix();
		//check();
		gpu_core_time = ins->core_time;
	}

	LOG(logger, "cpu_time = %lf s, gpu_time = %lf s, speed up = %.2lf\n", cpu_core_time / (double)CLOCKS_PER_SEC, gpu_core_time / (double)CLOCKS_PER_SEC, cpu_core_time / (double)gpu_core_time);

	return 0;
}
