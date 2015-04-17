#include "Demo.h"

#include <stdio.h>
#include <time.h>
#include <cuda.h>
#include <string.h>
#include <limits.h>

#include "./DocClustering/KMeansClustering.h"
#include "./DataSource/WikipediaDataSource.h"
#include "./Common/IDFManager.h"
#include "./DocClustering/KMeansClusteringCPU.h"
#include "./DocClustering/KMeansClusteringGPU.h"

static clock_t test(KMeansClustering *cluAlgo, DocumentSource *doc_src, int doc_num, int c) {
	doc_src->openSource();

	cluAlgo->initilize();
	for(int i=0;i < doc_num && doc_src->hasNext();i++) {
		if(i%100 == 0) {
			// printf("adding document %d\n", i+1);
		}
		cluAlgo->add_document(doc_src->getNextDocument().c_str());
	}
	printf("Add %d docs\n", doc_num);
// 	vector<int> ids;
// 	for(int i=130;i<160;i++) ids.push_back(i);
// 	cluAlgo->set_init_centroids(ids);
	clock_t ttt = clock();
	cluAlgo->run_clustering(c);
	clock_t ret = clock() - ttt;

	cluAlgo->print_result();
	cluAlgo->destroy();

	doc_src->closeSource();
	return ret;
}

static void print_usage() {
	printf("Document Clustering Usage\n");
	printf("-idf_file    indicate the idf-file name.\n");
	printf("-calc_idf    need to calc idf value, followed by top-words-num big-corpus lines. e.g. -calc_idf 20000 ./plain.txt 6000000");
	printf("-cpu         test in CPU\n");
	printf("-gpu         test in GPU\n");
	printf("-corpus      big text corpus, e.g. ./data/wiki_plain_txt\n");
	printf("-log         log file name, e.g. ./log.txt\n");
	printf("-docs_num    max documents, eg 6000000\n");
	printf("-centroids   indicate centroids.");
	exit(0);
}

int doc_clustering_test(int argc, char** argv) {
	Logger *file_logger = NULL;

	string idf_file = "";
	
	bool calc_idf = false;
	int top_words = 20000;
	string idf_doc_src;
    int idf_doc_num = 6000000;

	int mask = 0;
	string corpus = "";
	int doc_num = 6000000, centroids = 1; 
	for (int i = 2; i < argc;) {
		if (strcmp(argv[i], "-log") == 0 && i + 1 < argc) {
			file_logger = new Logger(argv[i + 1], false);
			i += 2;
		} else if (strcmp(argv[i], "-calc_idf") == 0 && i + 3 < argc) {
            calc_idf = true;
            top_words = atoi(argv[i + 1]);
			idf_doc_src = argv[i + 2];
            idf_doc_num = atoi(argv[i + 3]);
			i += 4;
		} else if (strcmp(argv[i], "-idf_file") == 0 && i + 1 < argc) {
			idf_file = argv[i + 1];
			i += 2;
		} else if (strcmp(argv[i], "-top_words") == 0 && i + 1 < argc) {
			top_words = atoi(argv[i + 1]);
			i += 2;
		} else if (strcmp(argv[i], "-corpus") == 0 && i + 1 < argc) {
			corpus = argv[i + 1];
			i += 2;
		} else if (strcmp(argv[i], "-cpu") == 0) {
			mask |= 1;
			i ++;
		} else if (strcmp(argv[i], "-gpu") == 0) {
			mask |= 2;
			i ++;
		} else if (strcmp(argv[i], "-doc_num") == 0 && i + 1 < argc) {
			doc_num = atoi(argv[i + 1]);
			i += 2;
		} else if (strcmp(argv[i], "-centroids") == 0 && i + 1 < argc) {
			centroids = atoi(argv[i + 1]);
			i += 2;
		} else {
			print_usage();
		}
	}
	Logger idf_logger(stdout);
	IDFManager idf_manager(&idf_logger);
	idf_manager.load_idf(idf_file.c_str());
	if (calc_idf) {
		WikipediaDataSource *doc_src = new WikipediaDataSource(idf_doc_src);
		doc_src->set_max_docs(idf_doc_num);
		idf_manager.calc_idf(doc_src, top_words, idf_file.c_str());
	}

	Logger logger(stdout, file_logger);
	clock_t cpu_time = 1, gpu_time = 1;

	// test in cpu
	if (mask & 1) {
		WikipediaDataSource *wiki_src = new WikipediaDataSource(corpus);
		wiki_src->set_max_docs(doc_num * 3);
		KMeansClusteringGPU *cpu = new KMeansClusteringCPU(&logger, &idf_manager);
		cpu_time = test(cpu, wiki_src, doc_num, centroids);
		delete cpu;
		delete wiki_src;
	}

	// test in gpu
	if (mask & 2) {
		WikipediaDataSource *wiki_src = new WikipediaDataSource(corpus);
		wiki_src->set_max_docs(doc_num * 3);
		KMeansClusteringCPU *gpu = new KMeansClusteringGPU(&logger, &idf_manager);
		gpu_time = test(gpu, wiki_src, doc_num, centroids);
		delete gpu;
		delete wiki_src;
	}
	
	printf("cpu_time = %lf s, gpu_time = %lf s, speed up: %lf\n", cpu_time / (double)CLOCKS_PER_SEC, gpu_time / (double)CLOCKS_PER_SEC, cpu_time / (double)gpu_time);
// 	if(gpu->clusters != cpu->clusters) {
// 		puts("Wrong!!!!!!!!"); 
// 	} else {
// 		puts("Yes...........");
// 	}

	return 0;
}
//18:53
