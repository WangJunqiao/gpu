//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file DataPoint.h
/// @brief Class DataPoint is representation of Vector Space Model of document.
///
/// In Document Clustering algorithm, we use Document Vector Model and a document is represented by a DataPoint.
/// The Inverted Document Frequency(IDF) of words is global map managed by class IDFManager.
///
/// @version 1.0
/// @author Junqiao Wang
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////

#ifndef DATA_POINT_H
#define DATA_POINT_H

#include <memory.h>

/// @brief Class DataPoint is representation of Vector Space Model of document.
///
/// In Document Clustering algorithm, we use Document Vector Model and a document is represented by a DataPoint.
/// The Inverted Document Frequency(IDF) of words is global map managed by class IDFManager.
class DataPoint{
public:
	
	int dim; 
	float *data; 

	DataPoint(const DataPoint& rh) {
		dim = rh.dim;
		data = (float *)malloc(sizeof(float) * dim);
		copy(rh.data, rh.data + dim, data);
	}

	DataPoint& operator = (const DataPoint& rh) {
		if (this == &rh) return *this;
		if (dim != rh.dim) {
			dim = rh.dim;
			if(data) free(data);
			data = (float*)malloc(dim * sizeof(float));
		}
		copy(rh.data, rh.data + dim, data);
		return *this;
	}

	DataPoint(int dimensions) {
		dim = dimensions;
		data = (float *)malloc(sizeof(float) * dim);
	}

	~DataPoint() {
		if(data) free(data);
	}
	DataPoint() {
		dim = 0;
		data = NULL;
	};
};

#endif