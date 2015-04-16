#include "WordSimCalcCPUImpl.h"

#include <vector>

#include "MatrixFileReader.h"
#include "../Common/Logger.h"

using namespace std;


CPUWordSimCalculator::CPUWordSimCalculator(Logger *logger, const string &result_dir, int top_words_num) 
	: WordSimCalculator(logger, result_dir, top_words_num) {
		//this->WordSimCalculator::WordSimCalculator(logger, matrix_file, matrix_file2, word_file);
}

void CPUWordSimCalculator::calc_similarity_matrix() {

	clock_t ttt = clock();

	Logger reader_logger(stdout);
	MatrixFileReader reader(&reader_logger);

	//return;

	vector<int> v = reader.init_reader(
		get_matrix_file_name(1).c_str(), 
		get_word_file_name().c_str()
		);

	//return;
    LOG(logger, "matrix_file_name2 = %s", get_matrix_file_name(2).c_str());
	FILE *fp = fopen(get_matrix_file_name(2).c_str(), "wb");
	
    clock_t t = clock();
	for(int id1=0;id1<(int)v.size();id1++) {
		if(id1 % 1000 == 0)
			LOG(logger, "vector %d completed, time used: %d ms", id1, (int)(clock()-t));
        int *i1, *i2;
		float *f1, *f2;
		reader.load_data(id1);
		i1 = reader.r_iptr[id1];
		f1 = reader.r_fptr[id1];
		//read_data(id1, &i1, &f1);
		for(int j=1;j<=i1[0];j++) {
			int id2 = i1[j];
			if(id2 < id1) 
				continue;
			reader.load_data(id2);
			i2 = reader.r_iptr[id2];
			f2 = reader.r_fptr[id2];
			//read_data(id2, &i2, &f2);

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
			double sf1 = 0, sf2 = 0;
			for(int i=1;i<=i1[0];i++) sf1 += f1[i];
			for(int i=1;i<=i2[0];i++) sf2 += f2[i];
			//float res = sum/(f1[0]+f2[0]);
			float res = sum / (sf1 + sf2);
			fwrite(&id1, sizeof(int), 1, fp);
			fwrite(&id2, sizeof(int), 1, fp);
			fwrite(&res, sizeof(float), 1, fp);
		}
    }
	fclose(fp);
	//return;

	core_time = clock() - ttt;

	rebuild_triples(2, 2);

	
	//printf("dddddddddddddddddddddddddddddddddddd\n");
}
