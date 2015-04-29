//This file is deprecated!!
#ifndef DOCGENERATOR_H
#define DOCGENERATOR_H

class DocGenerator {
public:
	/*
	Param dir: the directory where documents should be placed.
	Param N: number of document should be generated.
	Param minDocLength: minimal bytes a document contains.
	Param maxDocLength: maximal bytes a document contains.
	Param aveDupDocs: the average documents a document will be replicated with.
	Param variance: 
	*/
	virtual void generate(char *dir, int N, int minDocLength, int maxDocLength, double aveDupDocs, double variance) = 0;
};

#endif