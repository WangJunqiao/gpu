////////////////////////////////////////////////////////////////////
//////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file MatrixGen.h
/// @brief A @ref DataMatrix class to generate and maintain the data matrix
///
/// This file contains a @ref DataMatrix class, you can use @ref generate_matrix method to
/// generate the corpus data matrix, and you can use @ref generate_query_matrix to generate the
/// query data matrix, after you generating the matrix you can use a series of get methods to
/// get the information you want.
///
/// @version 1.0
/// @author Bowen Liu
/// @date 01.17.2014
///
////////////////////////////////////////////////////////////////////
//////
#ifndef _MATRIXGEN_H_
#define _MATRIXGEN_H_

#include <iostream>
#include "MyCommon.h"
#include "../Common/Logger.h"

/// @brief A DataMatrix class to generate data matrix
///
/// The objects of this class can generate corpus data matrix and query data matrix, note that
/// this class has copy construct method and operator=, this is because the object of this class
/// may need to assign value to each other
/// sample use: 
///			DataMatrix *pm = new DataMatrix(&logger);
///			pm->generate_matrix(data_path);
///			pm->generate_query_matrix(query_data_path);
///	then you can use get methods to get the information you want:
///			data = pm->get_data();
///			query_data = pm->get_query_data();
///
class DataMatrix {
public:
	/// @brief construct method
	///
	/// the construct method is the default constructor, it will set the integer values
	/// to 0 and pointer values to NULL
	DataMatrix(){
		doc_num = 0;
		doc_dim = 0;
		query_num = 0;
		data = NULL;
		query_data = NULL;
	}
	/// @brief construct method
	///
	/// the construct method is the default constructor, it will pass into a logger pointer
	///
	/// @param[in] logger: the Logger pointer to print log info.
	DataMatrix(Logger *logger){
		this->logger = logger;
		doc_num = 0;
		doc_dim = 0;
		query_num = 0;
		data = NULL;
		query_data = NULL;
	}
	/// @brief copy construct method
	///
	/// this method is mainly to handle the call by value and call by reference 
	///
	/// @param[in] m: the data matrix you want to copy.
	DataMatrix(const DataMatrix& m) {
		doc_num = m.doc_num;
		doc_dim = m.doc_dim;
		query_num = m.query_num;
		data = (DataType*)malloc(doc_num*doc_dim*sizeof(DataType));
		query_data = (DataType*)malloc(query_num*doc_dim*sizeof(DataType));
		memcpy(data, m.data, doc_num*doc_dim*sizeof(DataType));
		memcpy(query_data, m.query_data, query_num*doc_dim*sizeof(DataType));
	}
	DataMatrix &operator=(const DataMatrix& rhs) {
		if(this == &rhs){
			return *this;
		}
		doc_num = rhs.doc_num;
		doc_dim = rhs.doc_dim;
		query_num = rhs.query_num;
		data = (DataType*)malloc(doc_num*doc_dim*sizeof(DataType));
		query_data = (DataType*)malloc(query_num*doc_dim*sizeof(DataType));
		memcpy(data, rhs.data, doc_num*doc_dim*sizeof(DataType));
		memcpy(query_data, rhs.query_data, query_num*doc_dim*sizeof(DataType));
		return *this;
	}
	~DataMatrix(){
		if (data != NULL) delete[] data;
		if (query_data != NULL)delete[] query_data;
	}

	/// @brief  generate data matrix according to the tf-idf file
	/// 
	/// generate data matrix according to the tf-idf file, the tf-idf file
	/// should be formated as follow: <doc_id, word_id, tf-idf value>
	///
	/// @param[in] path: the path of tf-idf file;	
	/// @return true if matrix successfully generated 
	/// @note the data matrix is a full matrix
	bool generate_matrix(char *path);

	/// @brief  generate query data matrix according to the query tf-idf file
	/// 
	/// generate query data matrix according to the query tf-idf file, the tf-idf file
	/// should be formated as follow: <doc_id, word_id, tf-idf value>
	///
	/// @param[in] path: the path of query tf-idf file;	
	/// @return true if matrix successfully generated 
	/// @note the query data matrix is a full matrix
	bool generate_query_matrix(char *path);

	/// @brief  get the data matrix
	/// 
	/// get the data matrix, return the pointer to the matrix
	///
	/// @return the pointer to the matrix 
	DataType* get_data();

	/// @brief  get the query data matrix
	/// 
	/// get the query data matrix, return the pointer to the query matrix
	///
	/// @return the pointer to the query matrix
	DataType* get_query_data();

	/// @brief  get the document number of corpus
	///
	/// @return document number of corpus
	int get_num(void);

	/// @brief  get the document number of query data
	///
	/// @return document number of query data
	int get_query_num(void);

	/// @brief  get the dimension of documents
	///
	/// @return the dimension of documents
	/// @note the dimension of corpus and query data must be the same
	int get_dim(void);
	
private:
	int doc_num; //num of Docs
	int doc_dim; //dimension
	int query_num; //num of query Docs
	DataType *data;
	DataType *query_data;
	Logger *logger;
};

#endif