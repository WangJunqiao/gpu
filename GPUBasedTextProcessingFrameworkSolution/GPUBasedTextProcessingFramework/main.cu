#include "Demo.h"

#include <stdio.h>

void print_usage_main() {
	puts("Demo Usage   -options");
	puts("-word_sim    word similarity calculation");
	puts("-doc_dup     document duplicate detection");
}

int main(int argc, char** argv) {
	if (argc < 2) {
		print_usage_main();
		return 0;
	}
	if (strcmp(argv[1], "-word_sim") == 0) {
		word_similarity_test(argc, argv);
	} else {
		print_usage_main();
		return 0;
	}
	//doc_clustering_test();
	//doc_dup_detection_test();
	
	//	doc_knn_test();
	//testmul();
	//test_word_sim();
}