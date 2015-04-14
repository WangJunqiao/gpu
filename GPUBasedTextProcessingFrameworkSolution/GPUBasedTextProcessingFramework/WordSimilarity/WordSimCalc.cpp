#include "WordSimCalc.h"

#include <stdio.h>

#include <vector>
#include <string>
#include <cassert>
#include <cmath>
#include <ctime>
#include <algorithm>
#include <unordered_map>
#include <sstream>

#include "../Common/Common.h"
#include "../Common/Logger.h"

using namespace std;


WordSimCalculator::WordSimCalculator(Logger *logger, const string &result_dir, int top_words_num) {
	this->logger = logger;
	this->result_dir = result_dir;
	this->top_words_num = top_words_num;
	if(this->result_dir.back() != '/' && this->result_dir.back()!='\\') {
		this->result_dir += "/";
	}
}

string WordSimCalculator::get_word_file_name() {
	return this->result_dir + "word_file";
}

string WordSimCalculator::get_matrix_file_name(int order) {
	if(order < 0) {
		return this->result_dir + "matrix_tmp";
	} else if (order == 1) {
		return this->result_dir + "mutual_info_matrix";
	} else if (order == 2) {
		return this->result_dir + "similarity_matrix";
	} else {
		logger->printf("Wrong order in function get_matrix_file_name: %d\n", order);
		exit(1);
	}
}


class StringHasher{
public:
	size_t operator()(const string &s) const {
		size_t h = 0;
		for(int i=0;i<(int)s.length();i++) {
			h = (h*130 + s[i]) % 10000007;
		}
		return h;
	}
};

class PairSIComp{
public:
	//按照pair<string, int>的int从大到小排序
	bool operator()(const pair<string, int> &p1, const pair<string, int> &p2) const{
		if(p1.second > p2.second) 
			return true;
		return false;
	}
};

typedef unordered_map<string, int, StringHasher> HashMapSI;
typedef unordered_map<int, int, hash<int> > HashMapII;
typedef unordered_map<int, float, hash<int> > HashMapIF;

/*
将所有单词大小写归一化。一般单词小写，所有字母都大写的单词保持不变
*/
static void process_word(string &ch) {
	bool all_large = true;
	for(int i=0;i<(int)ch.length();i++) {
		if(ch[i]>='A' && ch[i]<='Z') ;
		else all_large = false;
	}
	if(!all_large) {
		for(int i=0;i<(int)ch.length();i++) {
			if(ch[i]>='A' && ch[i]<='Z') {
				ch[i] += ('a'-'A');
			}
		}
	}
}

//word_id记录一个字符串到id(一个base0的整数)的映射关系
static HashMapSI word_id;

//id_change表示word_id再次映射到排序后的id
//static HashMapII id_change;



//找出有意义的单词，在corpus文件中出现的次数不少于MIN_OCCURENCE次
void WordSimCalculator::find_top_words(DocumentSource *doc_src) {
	LOG(logger, "Finding top ranked(top %d) words...", this->top_words_num);
	HashMapSI counter;

	doc_src->openSource();
	int line = 0;
	while(doc_src->hasNext()) {
		istringstream sin(doc_src->getNextDocument());
		string word;
		while(sin>>word) {
			if(word.size() <= 1) {
				continue;
			}
			process_word(word);
			counter[word] ++;
		}
		line++;
		if(line%10000==0) {
			LOG(logger, "%d documents processed.", line);
		}
	}
	doc_src->closeSource();

	word_id.clear();
	vector<pair<string, int> > v(counter.begin(), counter.end());
	sort(v.begin(), v.end(), PairSIComp()); //按照出现的次数降序排列
	for(int i=0;i<this->top_words_num && i<(int)v.size();i++) {
		word_id[v[i].first] = i;
	}

	LOG(logger, "tot %d words, minimal occurrence = %d", (int)word_id.size(), v[word_id.size()-1].second);
}


void WordSimCalculator::calc_mutual_info_matrix(DocumentSource *doc_src, int win_size) {
	clock_t t = clock(), ttt = t;
	find_top_words(doc_src);
	int W = word_id.size();
	LOG(logger, "find top words, time used = %f s", (clock()-t)/(double)CLOCKS_PER_SEC);

	//cnt_map[i]记录第i个字符串跟那些字符串成对出现过，并统计次数
	vector<HashMapII*> cnt_map;
	//co_oc[i]表示第cnt_map[i]中所有pair的second总和
	vector<long long> co_oc;
	for(int i=0;i<word_id.size();i++) {
		cnt_map.push_back(new unordered_map<int, int, hash<int> >()); //freed
		co_oc.push_back(0);
	}
	//co_oc_s表示co_oc数组的总和
	long long co_oc_s = 0;

	doc_src->openSource();
	int line = 0, max_size = 0;	
	long long word_tot = 0;
	while(doc_src->hasNext()) {
		istringstream sin(doc_src->getNextDocument());
		vector<int> vs;
		string word;
		while(sin>>word) {
			word_tot ++;
			process_word(word);
			if(word_id.find(word) == word_id.end()) {
				vs.push_back(-1);
			} else {
				vs.push_back(word_id[word]); 
			}
		}
		for (int i = 0; i < (int)vs.size(); i ++) if(vs[i] >= 0) {
			for (int off = -win_size; off <= win_size; off ++) if(off != 0) {
				int j = i + off;
				if (j >= 0 && j < (int)vs.size() && vs[j] >= 0) {
					(*cnt_map[vs[i]])[vs[j]] ++;
				}
			}
			max_size = max(max_size, (int)cnt_map[vs[i]]->size());
		}

		line++;
		if (line%10000 == 0) {
			LOG(logger, "line = %d, max_size = %d", line, max_size);
		}
		//if(line>1000000) break;
	}

// 	for(HashMapSI::iterator it = word_id.begin();it!=word_id.end();++it) {
// 		v.push_back(make_pair(cnt_map[it->second]->size(), it->first));
// 	}
//	sort(v.begin(), v.end());
//	reverse(v.begin(), v.end());

// 	id_change.clear();
// 	for(int i=0;i<(int)v.size();i++) {
// 		id_change[word_id[v[i].first]]= i; //根据单词的vector长度来排序，越长的编号越小
// 	}

	LOG(logger, "word_tot = %lld", word_tot);

	FILE *fout = fopen(get_matrix_file_name(1).c_str(), "wb");
	for(int i=0;i<W;i++) {
		co_oc[i] = 0;
		for(HashMapII::iterator it = cnt_map[i]->begin();it!=cnt_map[i]->end();++it) {
			co_oc[i] += it->second;
		}
		co_oc_s += co_oc[i];
	}
	double tt = log((double)co_oc_s), max_val = 0;
	vector<long long> pos;
	for(int i=0;i<W;i++) {
		vector<pair<int, float> > vif;
		int num = 0;
		float sum = 0.0;
		for(HashMapII::iterator it = cnt_map[i]->begin();it!=cnt_map[i]->end();++it) {
			
			int tid = it->first;
			if(tid == i) continue;
			double val = tt;
			val += log((double)it->second);
			val -= log((double)co_oc[i]);
			val -= log((double)co_oc[it->first]);

			if(val <= 0) {
				continue; //只考虑大于0的
				LOG(logger, "bad case");
			}
			max_val = max(max_val, val);

			num ++;
			sum += (float)val;

			vif.push_back(make_pair(tid, (float)val));
		}
		pos.push_back(ftell(fout));
		fwrite(&num, sizeof(int), 1, fout);
		fwrite(&sum, sizeof(float), 1, fout);
		sort(vif.begin(), vif.end());
		for(int j=0;j<(int)vif.size();j++) {
			fwrite(&vif[j].first, sizeof(int), 1, fout);
			fwrite(&vif[j].second, sizeof(float), 1, fout);
		}
	}
	printf("max_val = %f\n", max_val);

	FILE *fp = fopen(get_word_file_name().c_str(), "w");
	vector<string> words(word_id.size(), "");
	for(HashMapSI::iterator it = word_id.begin();it!=word_id.end();++it) {
		words[it->second] = it->first;
	}
	for(int i=0;i<W;i++) {
		string wd = words[i];
		fprintf(fp, "%d %s %lld %d\n", i, wd.c_str(), pos[i], cnt_map[i]->size());
	}
	fclose(fp);
	fclose(fout);

	for(int i=0;i<cnt_map.size();i++) {
		delete cnt_map[i];
	}

	LOG(logger, "tot time: %f s", (clock()-ttt)/(double)CLOCKS_PER_SEC);
}



void WordSimCalculator::rebuild_triples(int order1, int order2) {
	vector<HashMapIF*> sims(word_id.size(), NULL);
	for(int i=0;i<sims.size();i++) {
		sims[i] = new HashMapIF(); // freed 
	}

	FILE *fp = fopen(get_matrix_file_name(order1).c_str(), "rb");
	int id1, id2, cnt = 0;
	float val;
	while(fread(&id1, sizeof(int), 1, fp) > 0) {
		fread(&id2, sizeof(int), 1, fp);
		fread(&val, sizeof(float), 1, fp);
		sims[id1]->insert(make_pair(id2, val));
		sims[id2]->insert(make_pair(id1, val));
		cnt ++;
	}
	fclose(fp);
	LOG(logger, "rebuild %d pairs", cnt);

	fp = fopen(get_matrix_file_name(order2).c_str(), "wb");
	for(int i=0;i<word_id.size();i++) {
		vector<pair<int, float> > vif(sims[i]->begin(), sims[i]->end());
		int num = vif.size();
		float sum = 0.0;
		for(int j=0;j<(int)vif.size();j++) {
			sum += vif[j].second;
		}
		//pos.push_back(ftell(fout));
		fwrite(&num, sizeof(int), 1, fp);
		fwrite(&sum, sizeof(float), 1, fp);
		sort(vif.begin(), vif.end());
		for(int j=0;j<(int)vif.size();j++) {
			fwrite(&vif[j].first, sizeof(int), 1, fp);
			fwrite(&vif[j].second, sizeof(float), 1, fp);
		}
		if(i%1000 == 0) {
			LOG(logger, "rebuild %d", i);
		}
	}
	fclose(fp);

	for(int i=0;i<sims.size();i++) {
		delete sims[i];
	}
	LOG(logger, "rebuild successfully");
}