#include "Demo.h"

#include <stdio.h>
#include <time.h>
#include <cuda.h>

#include "./DocClustering/KMeansClustering.h"
#include "./DataSource/WikipediaDataSource.h"
#include "./Common/IDFManager.h"
#include "./DocClustering/KMeansClusteringCPU.h"
#include "./DocClustering/KMeansClusteringGPU.h"

static void test(KMeansClustering *cluAlgo, DocumentSource *doc_src) {
	doc_src->openSource();

	cluAlgo->initilize();
	for(int i=0;i<200 && doc_src->hasNext();i++) {
		if(i%100 == 0) 
			printf("adding document %d\n", i+1);
		cluAlgo->add_document(doc_src->getNextDocument().c_str());
	}
// 	vector<int> ids;
// 	for(int i=130;i<160;i++) ids.push_back(i);
// 	cluAlgo->set_init_centroids(ids);
	cluAlgo->run_clustering(30);

	cluAlgo->print_result();
	cluAlgo->destroy();

	doc_src->closeSource();
}

int doc_clustering_test() {

	Logger idf_logger(stdout);

	WikipediaDataSource *wiki_src = new WikipediaDataSource("./data/wiki_plain_txt");
	IDFManager idf_manager(&idf_logger);
	idf_manager.calc_idf(wiki_src, 20000, "./data/word_idf");

	//Logger cpu_logger(stdout);
	Logger cpu_logger("./data/cpu_log.txt", false);
	Logger gpu_logger("./data/gpu_log.txt", false);
	
	KMeansClusteringGPU *gpu = new KMeansClusteringGPU(&gpu_logger, &idf_manager);
	KMeansClusteringCPU *cpu = new KMeansClusteringCPU(&cpu_logger, &idf_manager);

	test(gpu, wiki_src);
	test(cpu, wiki_src);

// 	if(gpu->clusters != cpu->clusters) {
// 		puts("Wrong!!!!!!!!"); 
// 	} else {
// 		puts("Yes...........");
// 	}

	delete gpu;
	delete cpu;
	return 0;
}
//18:53
