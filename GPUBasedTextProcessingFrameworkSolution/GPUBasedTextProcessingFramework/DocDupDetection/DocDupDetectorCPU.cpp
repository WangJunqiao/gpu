#include "DocDupDetectorCPU.h"

#include "../Common/Common.h"
#include <iostream>
#include <string>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <list>
#include <map>

using namespace std;

#define ROLLING_WINDOW 7
#define HASH_PRIME 0x01000193
#define IGNORE_WHITE_SPACE 1
#define SPAMSUM_LENGTH 64 
#define MIN_BLOCKSIZE 3
#define HASH_INIT 0x28021967
#define judge_score 80
typedef unsigned u32;
typedef unsigned char uchar;

list<int> L;

struct {
	uchar window[ROLLING_WINDOW]; 
	u32 h1,h2,h3;
	u32 n;
}roll_state;

//滚动哈希计算
/*typedef struct Node{
int doc_id;
char hash[150];
} Pair;*/
//vector<Pair> v;
map<int, string> M;
map<int, string> docs;
//vector<pair<int,int> > candidate;
map<int, vector<int> > allDupPair;
map<int, vector<int> > afterRefineCandPairs;
int doc_id=0;

u32 DocDupDetectorCPU::roll_hash(uchar c){
	roll_state.h2 -= roll_state.h1;
	roll_state.h2 += ROLLING_WINDOW * c;

	roll_state.h1 += c;
	roll_state.h1 -= roll_state.window[roll_state.n % ROLLING_WINDOW];

	roll_state.window[roll_state.n % ROLLING_WINDOW] = c;
	roll_state.n++;

	roll_state.h3 = (roll_state.h3 << 5) & 0xFFFFFFFF;
	roll_state.h3 ^= c;

	return roll_state.h1 + roll_state.h2 + roll_state.h3;
}

u32 DocDupDetectorCPU::roll_reset(void){	
	memset(&roll_state, 0, sizeof(roll_state));
	return 0;
}

u32 DocDupDetectorCPU::sum_hash(uchar c, u32 h){
	h *= HASH_PRIME;
	h ^= c;
	return h;
}

char* DocDupDetectorCPU::spamsum(const uchar *in, int length, int flags, u32 bsize) {
	const char *b64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
	char *ret,*p;
	int total_chars;
	u32 block_size;
	u32 h,h3,h2;
	int j,k;
	char ret2[SPAMSUM_LENGTH/2+1];
	if(flags&IGNORE_WHITE_SPACE){
		int n,i;
		for(n=0,i=0;in[i];i++){
			if(isspace(in[i]))
				continue;
			n++;
		}
		total_chars=n;
	} else {
		total_chars=length;
	}
	if(bsize==0) {
		block_size = MIN_BLOCKSIZE;
		while(SPAMSUM_LENGTH*block_size<total_chars){
			block_size=block_size*2;
		}
	} else {
		block_size=bsize;
	}
	ret=(char*)malloc(SPAMSUM_LENGTH+SPAMSUM_LENGTH/2+20);
	//memset(ret,'\0',sizeof(ret));
	if(!ret)
		return 0;
	int count=0;
	do{
		count++;
		if(count>1)
			block_size=block_size/2;
		_snprintf(ret, 12, "%u:", block_size);  //12可以放大
		//printf("%s\n",ret);
		//p=(char*)malloc(SPAMSUM_LENGTH+1);
		p=ret+strlen(ret);
		memset(p,'\0',sizeof(p));
		memset(ret2,'\0',sizeof(ret2));
		h=roll_reset();
		h2=HASH_INIT;
		h3=HASH_INIT;
		j=0,k=0;
		for(int i=0;i<length;i++){
			if((flags&IGNORE_WHITE_SPACE)&&isspace(in[i]))
				continue;
			h=roll_hash(in[i]);
			h2=sum_hash(in[i],h2);
			h3=sum_hash(in[i],h3);
			if(h%block_size==(block_size-1)){
				p[j]=b64[h2%64];
				p[j+1]='\0';
				if(j<SPAMSUM_LENGTH-1){
					j++;
					h2=HASH_INIT;
				}
			}
			if(h%2*block_size==(2*block_size-1)){
				ret2[k]=b64[h3%64];
				ret2[k+1]='\0';
				if(k<SPAMSUM_LENGTH/2-1){
					h3=HASH_INIT;
					k++;
				}
			}
		}
		if(h){
			p[j]=b64[h2%64];
			ret2[k]=b64[h3%64];
			p[j+1]='\0';
			ret2[k+1]='\0';
		}
		strcat(p+j,":");
		strcat(p+j,ret2);
	}while(bsize==0&&block_size>MIN_BLOCKSIZE&&j<SPAMSUM_LENGTH/2);
	return ret;
}

int DocDupDetectorCPU::check_common_substring(const char* hash_value1, const char* hash_value2){
	int len1=strlen(hash_value1);
	int len2=strlen(hash_value2);
	int i,j;
	int max_len=0;
	for(i=0;i<len1;i++){
		char ch=hash_value1[i];
		for(j=0;j<len2;j++){
			if(hash_value2[j]==ch){
				int k,t;
				int loc_len=0;
				for(k=i,t=j;k<len1&&t<len2;k++,t++){
					if(hash_value1[k]==hash_value2[t])
						loc_len++;
					else
						break;
				}
				if(loc_len>max_len)
					max_len=loc_len;
			}
		}
	}
	if(max_len>=5)
		return 1;
	else
		return 0;
}

void DocDupDetectorCPU::initialize() {
	doc_id = 0;
	M.clear();
	docs.clear();
	allDupPair.clear();
	afterRefineCandPairs.clear();
}

void DocDupDetectorCPU::add_document(string doc){
	int length=doc.length();
	int flags=0;
	u32 bsize=0;
	char *hash_value=spamsum((uchar*)doc.c_str(), length, flags, bsize);
	int id=doc_id;
	LOG(logger, "%d %s", doc_id, hash_value);
	docs[doc_id] = doc;
	M[doc_id++]=hash_value;
	//	return id;
}

void DocDupDetectorCPU::calculate_dups(){
	clock_t tt = clock();
	LOG(logger, "Begin calculate candidate dups");
	map<int,string>::iterator iter,next;
	for(iter=M.begin();iter!=M.end();iter++){
		const char* source_hash=(iter->second).c_str();
		int id=iter->first;
		for(next=M.begin();next!=M.end();next++){
			if(next->first==id)
				continue;
			const char* target_hash=(next->second).c_str();
			//if(check_common_substring(source_hash, target_hash))
			if(score(source_hash, target_hash) >= 60)
				(allDupPair[id]).push_back(next->first);
		}
		LOG(logger, "calculate_candy_dups %d completely", iter->first);
	}
	core_time = clock() - tt;
	LOG(logger, "Calculate candidate dups completely, time used = %d ms", clock()-tt);
}
//[2014-03-05 13:07:27] - Calculate candidate dups completely, time used = 1258227 ms

vector<int> DocDupDetectorCPU::get_candidate_dup_docs(int id){
	return allDupPair[id];
}

void DocDupDetectorCPU::refine(){
	map<int,vector<int> >::iterator iter;
	for(iter=allDupPair.begin();iter!=allDupPair.end();iter++){
		int doc_id=iter->first;
		const char* doc1=docs[doc_id].c_str();
		vector<int> item = allDupPair[doc_id];
		for(int i=0;i<item.size();i++){
			const char *doc2=docs[item[i]].c_str();
			if(score(doc1, doc2) >= judge_score){
				afterRefineCandPairs[doc_id].push_back(item[i]);
			}
		}
		LOG(logger, "refine doc_id = %d ended, candidates = %d, real_dups = %d", doc_id, (int)item.size(), afterRefineCandPairs[doc_id].size());
	}
}

vector<int> DocDupDetectorCPU::get_real_dup_docs(int id){
	return afterRefineCandPairs[id];
}








