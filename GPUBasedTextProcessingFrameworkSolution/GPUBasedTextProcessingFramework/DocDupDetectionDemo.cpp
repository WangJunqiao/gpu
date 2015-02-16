#include "Demo.h"

#include <iostream>
#include <vector>

#include "./DataSource/DocumentSource.h"
#include "./DocDupDetection/DocDupDetector.h"
#include "./DocDupDetection/DocDupDetectorCPU.h"
#include "./DocDupDetection/DocDupDetectorGPU.h"
#include "./DataSource/CodeForcesDataSource.h"

using namespace std;

#define MAX_DUP_DOCUMENTS 2000

void docDupDetectorTest(DocDupDetector *detector, CodeforcesDataSource *dataSource) {
	detector->initialize();

	dataSource->set_files_directory("d:/dd/codeforce-code");
	dataSource->openSource();
	
	int cnt = 0;
	while(dataSource->hasNext()) {
		detector->add_document(dataSource->getNextDocument());
		cnt ++;
		if(cnt == MAX_DUP_DOCUMENTS)
			break;
	}
	detector->calculate_dups();
	//detector->refine();

	int cas = 10;
	FILE *fp = stdout;
	assert(fp != NULL);
	while(cas--) {
		int did = rand() * rand() % MAX_DUP_DOCUMENTS;
		vector<int> candies = detector->get_candidate_dup_docs(did);
		string doc_name = dataSource->getDocumentName(did);
		//cout<<doc_name<<endl;
		fprintf(fp, "candidates for document %d[%s]: ", did, doc_name.c_str());
		for(int i=0;i<(int)candies.size();i++) {
			fprintf(fp, " %d[%s]", candies[i], dataSource->getDocumentName(candies[i]).c_str());
		}
		fprintf(fp, "\n");
	}
	//fclose(fp);
}



int doc_dup_detection_test() {
	Logger logger (stdout);
	CodeforcesDataSource *cf = new CodeforcesDataSource();
	DocDupDetector *det;

	det = new DocDupDetectorCPU(&logger);
	docDupDetectorTest(det, cf);
	clock_t cpu_time = det->core_time;

	printf("Test cpu doc dup dector complete\n");

	delete det;
	det = new DocDupDetectorGPU(&logger);
	docDupDetectorTest(det, cf);
	clock_t gpu_time = det->core_time;


	printf("cpu_time = %d ms, gpu_time = %d ms, speed up = %.2lf\n", cpu_time, gpu_time, cpu_time / (double)gpu_time);
	
	return 0;
}