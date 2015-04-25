#include "Demo.h"

#include <iostream>
#include <vector>

#include "./DataSource/DocumentSource.h"
#include "./DocDupDetection/DocDupDetector.h"
#include "./DocDupDetection/DocDupDetectorCPU.h"
#include "./DocDupDetection/DocDupDetectorGPU.h"
#include "./DataSource/CodeForcesDataSource.h"

using namespace std;

void docDupDetectorTest(DocDupDetector *detector, string files_dir, int max_doc, Logger *logger) {
	detector->initialize();

	CodeforcesDataSource *dataSource = new CodeforcesDataSource();
	dataSource->set_files_directory(files_dir);
	dataSource->openSource();
	
	int N;
	for(N = 0; dataSource->hasNext() && N < max_doc; ++ N) {
		detector->add_document(dataSource->getNextDocument());
	}
	detector->calculate_dups();
	//detector->refine();

	int cas = 10;
	while(cas--) {
		int did = rand() * (long long)rand() % N;
		vector<int> candies = detector->get_candidate_dup_docs(did);
		string doc_name = dataSource->getDocumentName(did);
		//cout<<doc_name<<endl;
		LOG(logger, "candidates for document %d[%s]: ", did, doc_name.c_str());
		for(int i=0;i<(int)candies.size();i++) {
			LOG(logger, " %d[%s]", candies[i], dataSource->getDocumentName(candies[i]).c_str());
		}
		//fprintf(fp, "\n");
	}
	//fclose(fp);
}

static void print_usage() {
	printf("Document Duplication Detection Usage\n");
	printf("-files_dir   define where the text files store.\n");
	printf("-cpu         test in CPU\n");
	printf("-gpu         test in GPU\n");
	printf("-log         log file name, e.g. ./log.txt\n");
	printf("-max_docs    max documents, eg 6000000\n");
	printf("-set_param   set blocks, threads, method");
	exit(0);
}

int doc_dup_detection_test(int argc, char** argv) {
	Logger *file_logger = NULL;
	string files_dir = ".";
	int mask = 0;
	int max_doc = 50000; 
	
	int blocks = -1, threads, method;
	for (int i = 2; i < argc;) {
		if (strcmp(argv[i], "-log") == 0 && i + 1 < argc) {
			file_logger = new Logger(argv[i + 1], false);
			i += 2;
		} else if (strcmp(argv[i], "-files_dir") == 0 && i + 1 < argc) {
			files_dir = argv[i + 1];
			i += 2;
		} else if (strcmp(argv[i], "-cpu") == 0) {
			mask |= 1;
			i ++;
		} else if (strcmp(argv[i], "-gpu") == 0) {
			mask |= 2;
			i ++;
		} else if (strcmp(argv[i], "-max_doc") == 0 && i + 1 < argc) {
			max_doc = atoi(argv[i + 1]);
			i += 2;
		} else if (strcmp(argv[i], "-set_param") == 0 && i + 3 < argc) {
			blocks = atoi(argv[i + 1]);
			threads = atoi(argv[i + 2]);
			method = atoi(argv[i + 3]);
			i += 4;
		} else {
			print_usage();
		}

	}
	Logger *logger = new Logger(stdout, file_logger);
	
	clock_t cpu_time = 1, gpu_time = 1;

	if (mask & 1) {
		DocDupDetector *det = new DocDupDetectorCPU(logger);
		docDupDetectorTest(det, files_dir, max_doc, logger);
		cpu_time = det->core_time;
		LOG(logger, "%s", "Test cpu doc dup dector complete");
		delete det;
	}

	if (mask & 2) {
		DocDupDetectorGPU *det = new DocDupDetectorGPU(logger);
		if (blocks != -1) {
			det->set_param(blocks, threads, method);
		}
		docDupDetectorTest(det, files_dir, max_doc, logger);
		gpu_time = det->core_time;
		delete det;
	}

	LOG(logger, "cpu_time = %lf s, gpu_time = %lf s, speed up = %.2lf\n", cpu_time / (double)CLOCKS_PER_SEC, gpu_time / (double)CLOCKS_PER_SEC, cpu_time / (double)gpu_time);
	
	return 0;
}
