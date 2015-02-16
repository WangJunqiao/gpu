//////////////////////////////////////////////////////////////////////////
/// COPYRIGHT NOTICE
/// Copyright (c) 2014, Zhejiang University, Database Laboratory
/// All rights reserved.
///
/// @file DocumentSource.h
/// @brief A virtual class defines some common function of a document source, which is widely used in the toolkit. 
///
/// This is a virtual class. You should define a new class which extends this class, if you want to process your own data.
///
/// @version 1.0
/// @author Junqiao Wang
/// @date 01.15.2014
///
//////////////////////////////////////////////////////////////////////////

#ifndef DOCUMENTSOURCE_H
#define DOCUMENTSOURCE_H

#include <string>

/// @brief A virtual class defines some common function of a document source, which is widely used in the toolkit. 
///
/// This is a virtual class. You should define a new class which extends this class, if you want to process your own data.
class DocumentSource {
public:

	/// @brief Open this document source.
	///
	/// In this function, the document source will do some initialize work.
	///
	/// @return Success(true) or failure(false)
	virtual bool openSource() = 0;


	/// @brief Check whether there exists the next document.
	///
	/// 
	/// 
	/// @return exist(true) or not(false)
	virtual bool hasNext() = 0;


	/// @brief Fetch the next document.
	///
	/// 
	///
	/// @return the content of the next document, which is some concatenated words separated by space.
	virtual std::string getNextDocument() = 0;


	/// @brief Close the document source, release resource.
	///
	/// In this function, the document source will do some destructor work.
	///
	/// @return NULL.
	virtual void closeSource() = 0;


	virtual ~DocumentSource() {};
};

#endif