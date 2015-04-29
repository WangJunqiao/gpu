/*CpuDocGenerator *generator;
generator=new CpuDocGenerator();
char *dir;
dir=(char*)malloc(sizeof(char)*100);
scanf("%s",dir);
generator->generate(dir,10,20,40,3.5,2);   //第一个参数是需要产生文本的目录如：E:\myprojects\test  第二个参数是需要产生的文件数
第三个参数是最短的文本长度（字符），第四个从参数是最大的文本长度，第五个参数是每篇文章平均需要的重复数，第六个是方差
return 0;

*/
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include "DocGenerator.h"
#include <vector>
#include <list>
#include <string>
#include <string.h>
#include <algorithm>
#include <ctime>

using namespace std;

#define numOfCenterWords 10  //根据文章的长度决定需要的重点词，即需要重复的单词
#define maxNumOfPerCenterWord 5
#define percentOfmodify 0.2


int visited[200000];
//vector<list<int>> docs;
vector<vector<string>> contentPerdoc;
class CpuDocGenerator:public DocGenerator{
public: vector<string> words;
public:
	void initAllWords(char *dir);
	void generate(char *dir, int N, int minDocLength, int maxDocLength, double aveDupDocs, double variance);
	double Gaussion(double aveDupDocs,double variance);
	int randDocLength(int minDocLength,int maxDocLength);
	int randPick(int rand_x);
	vector<string> generateDupDoc(vector<string> content,int count);
	void generateFile(char *dir,int id,vector<string> content);
	void delay(int seconds);
//	void produceDoc(char* dir);
//	void docDup(int num,int id,int N,vector<string> dupwords,list<int>count);
};
void CpuDocGenerator::generate(char *dir, int N, int minDocLength, int maxDocLength, double aveDupDocs, double variance){
	int numberOfDup;
	int lengthOfDoc;
	char *filePath;
	filePath=(char*)malloc(sizeof(char)*100);
	//E:\myprojects\test\words\words
	scanf("%s",filePath);
	initAllWords(filePath);
	int numOfTotWords=words.size();
	vector<string> reserveDoc;
	for(int i=1;i<=N;i++){
		if(visited[i])
			continue;
		else{
		double value=Gaussion(aveDupDocs,variance);
		if(value<0){
			value-=0.5;
			numberOfDup=(int)abs(value);
		}
		else{
			value+=0.5;
			numberOfDup=(int)value;
		}
		lengthOfDoc=randDocLength(minDocLength,maxDocLength);
		int count=0;
			for(int j=1;count<lengthOfDoc&&j<=numOfCenterWords;j++){
			//	_sleep(700);
				int index=randPick(numOfTotWords);
				int dupCount=randPick(maxNumOfPerCenterWord);
				cout<<words[index]<<" "<<dupCount<<endl;
				for(int k=1;k<=dupCount;k++){
				 reserveDoc.push_back(words[index]);
				 random_shuffle(reserveDoc.begin(),reserveDoc.end());
				}
				count+=dupCount;
			}
		while(count<lengthOfDoc){
		//	_sleep(700);
			int index=randPick(numOfTotWords);
			count++;
			reserveDoc.push_back(words[index]);
			cout<<words[index]<<endl;
			random_shuffle(reserveDoc.begin(),reserveDoc.end());
		}
		visited[i]=1;
		for(int ii=0;ii<reserveDoc.size();ii++)
			printf("%s ",reserveDoc[ii].c_str());
		generateFile(dir,i,reserveDoc);
		for(int j=1;j<numberOfDup;j++){
			vector<string> newDoc;
		//	_sleep(700);
			int newId=randDocLength(i+1,N);
			if(!visited[newId]){
			newDoc=generateDupDoc(reserveDoc,numOfTotWords);
			generateFile(dir,newId,newDoc);
			visited[newId]=1;
			}
		}
		}
	}
	free(filePath);

}
vector<string> CpuDocGenerator::generateDupDoc(vector<string> content,int count){
	int length=content.size();
	int NumOfModify=(int)length*percentOfmodify;
	int i=1;
	vector<string> newDoc=content;
	while(i<NumOfModify){
		int ReplaceIndex=randPick(length);
		int wordIndex=randPick(count);
		newDoc[ReplaceIndex]=words[wordIndex];
		i++;
	}
	return newDoc;
}
void CpuDocGenerator::generateFile(char *dir,int id,vector<string> content){
	char filepath[50];
	strcpy(filepath,dir);
	sprintf(filepath,"%s%s",filepath,"\\");
	sprintf(filepath,"%s%d",filepath,id);
	sprintf(filepath,"%s%s",filepath,".txt");
	FILE *fp;
	fp=fopen(filepath,"w");
	if(fp==NULL)
		printf("Could not open the file!\n");
	else{
		int count=0;
		for(int i=0;i<content.size();i++){
			count++;
			if(count==10){
//				cout<<content[i].c_str()<<endl;
			  fprintf(fp,"%s",content[i].c_str());
			  fputc(10,fp);
			  count=0;
			}
			else{
//				cout<<content[i].c_str()<<endl;
				fprintf(fp,"%s ",content[i].c_str());
			}
		}
	}

}
double CpuDocGenerator::Gaussion(double aveDupDocs,double variance){
	static double V1, V2, S;
	static int phase = 0;
	double X;

	if ( phase == 0 ) {
		do {
			double U1 = (double)rand() / RAND_MAX;
			double U2 = (double)rand() / RAND_MAX;

			V1 = 2 * U1 - 1;
			V2 = 2 * U2 - 1;
			S = V1 * V1 + V2 * V2;
		} while(S >= 1 || S == 0);

		X = V1 * sqrt(-2 * log(S) / S);
	} else
		X = V2 * sqrt(-2 * log(S) / S);

	phase = 1 - phase;
	X=X*variance+aveDupDocs;
	return X;
}
void CpuDocGenerator::initAllWords(char *dir){
	ifstream infile;
	infile.open(dir);
	string temp;
	while(getline(infile,temp)){
	//	cout<<temp<<endl;
		words.push_back(temp);
	}
	memset(visited,0,sizeof(visited));
	srand((unsigned int)(time(NULL)));
}
int CpuDocGenerator::randDocLength(int start,int end){
	int temp,c;
	if(start>end){
		temp=start;
		start=end;
		end=temp;
	}
	c=end-start;
//	srand((unsigned)time(NULL));
	return rand()%c+start;
}

int CpuDocGenerator::randPick(int rand_x){
	//int numOfTotWords=words.size();
	//srand((unsigned int)(time(NULL)));
	return rand()%rand_x;
}

void CpuDocGenerator::delay(int seconds)  

{  

	clock_t start = clock();  

	clock_t lay = (clock_t)seconds * CLOCKS_PER_SEC;  

	while ((clock()-start) < lay)  

		;  

}  
