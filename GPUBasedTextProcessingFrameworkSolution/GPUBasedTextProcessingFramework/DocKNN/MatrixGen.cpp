#include "MatrixGen.h"
#include <iostream>
#include <assert.h>
using namespace std;

DataType* DataMatrix::get_data(){
	return data;
}

DataType* DataMatrix::get_query_data() {
	return query_data;
}

int DataMatrix::get_num(){
	return doc_num;
}

int DataMatrix::get_query_num(){
	return query_num;
}

int DataMatrix::get_dim(){
	return doc_dim;
}

bool DataMatrix::generate_matrix(char *tf_idf_path) {
	FILE *fp = fopen(tf_idf_path, "r");
	assert(fp != NULL);
	int tdoc_dim;
	fscanf(fp, "%d%d", &doc_num, &tdoc_dim);
	if (doc_dim != 0)assert(tdoc_dim == doc_dim);
	else doc_dim = tdoc_dim;
	assert(doc_num > 0 && doc_dim > 0);
	this->data = (DataType*)malloc(doc_num*doc_dim*sizeof(DataType));
	assert(this->data != NULL);
	memset(data, 0, doc_num*doc_dim*sizeof(DataType));
	int doc_id, word_id;
	DataType tf_idf;
	int count = 0;
	LOG(logger, "begin to read data...");
	while(fscanf(fp, "%d%d%lf", &doc_id, &word_id, &tf_idf) != EOF){
		assert(doc_id >= 0 && doc_id < doc_num);
		assert(word_id >= 0 && word_id < doc_dim);
		this->data[doc_id*doc_dim + word_id] = tf_idf;
		if (count++ % 1000 == 0)LOG(logger, "read %d terms.", count);
	}
	fclose(fp);
	return true;
}

bool DataMatrix::generate_query_matrix(char *tf_idf_path) {
	FILE *fp = fopen(tf_idf_path, "r");
	if(fp == NULL)return false;
	int query_dim;
	fscanf(fp, "%d%d", &query_num, &query_dim);
	if (doc_dim != 0)assert(query_dim == doc_dim);
	else doc_dim = query_dim;
	assert(query_num > 0 && query_dim > 0);
	query_data = (DataType*)malloc(query_num*query_dim*sizeof(DataType));
	memset(query_data, 0, query_num*query_dim*sizeof(DataType));
	int query_doc_id, query_word_id;
	DataType query_tf_idf;
	int count = 0;
	LOG(logger, "begin to read query data...");
	while(fscanf(fp, "%d%d%lf", &query_doc_id, &query_word_id, &query_tf_idf) != EOF){
		assert(query_doc_id >= 0 && query_doc_id < query_num);
		assert(query_word_id >= 0 && query_word_id < query_dim);
		query_data[query_doc_id*query_dim + query_word_id] = query_tf_idf;
		if (count++ % 1000 == 0)LOG(logger, "read %d terms.", count);
	}
	fclose(fp);
	return true;
}

