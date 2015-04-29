/*
DISCO单词相似度实验的预处理corpus得到一阶相似矩阵的程序
*/
#include <iostream>
#include <vector>
#include <string>
#include <cassert>
#include <stdio.h>
#include <cmath>
#include <ctime>
#include <algorithm>

using namespace std;


#define MAX_WORD_NUMBER 10000000


//K判相等用了==操作符
template<typename K, typename V, int(*hash)(const K &k)>
struct HashMap {
	struct Entry {
		K k;
		V v;
		int _h;
	} **l;
	int _size, _cap;
	double _factor;
	const static int PERCENT = 50; //当元素百分比超过多少时扩张
	const static int MUL = 4; //扩张的倍数，这个值必须是2的倍数

	HashMap(int capcity = 0) {
		_factor = PERCENT/100.0;
		_cap = 8;
		while(_cap * _factor < capcity) _cap <<= 1;
		_init(_cap);
	}

	HashMap(const HashMap &map) {
		*this = map;
	}

	HashMap& operator=(const HashMap &map) {
		_free();
		_init(map._cap);
		for(int i=0;i<_cap;i++) if(map.l[i]) {
			l[i] = new Entry();
			*l[i] = *map.l[i];
		}
		_size = map._size;
		return *this;
	}

	void _free() {
		for(int i=0;i<_cap;i++) if(l[i]) delete l[i];
		free(l);
	}

	int _geth(const K &k) {
		int h = hash(k);
		h ^= (h >> 20) ^ (h >> 12);
		return h ^ (h >> 7) ^ (h >> 4);
	}

	void _init(int cap) {
		_cap = cap;
		l = (Entry**) malloc(cap * sizeof (Entry*));
		_size = 0;
		for(int i=0;i<cap;i++) l[i] = NULL;
	}

	int _find(const K &k) { //return index
		int h = _geth(k);
		for (int i = h & (_cap-1); l[i]; i = (i+1)&(_cap-1))
			if (l[i]->_h==h && l[i]->k==k) return i;
		return -1;
	}

	int _insert(const K &k, const V &v) { //return index
		int i = _find(k);
		if (i >= 0) {
			l[i]->v = v;
			return i;
		}
		_size ++;
		int h = _geth(k);
		for (i = h & (_cap-1); l[i]; i = (i+1)&(_cap-1))
			if (l[i]->_h==h && l[i]->k==k) break;
		l[i] = new Entry();
		l[i]->k = k;
		l[i]->v = v;
		l[i]->_h = h;

		if(_size > _cap * _factor) {
			_rebuild();
			return _find(k);
		} else {
			return i;
		}
	}

	void _rebuild() {
		int ocap = _cap;
		_cap *= MUL;
		_size = 0;
		Entry** tl = l;
		l = (Entry**) malloc(_cap * sizeof (Entry*));
		for(int i=0;i<_cap;i++) l[i] = NULL;
		for(int i=0;i<ocap;i++) if(tl[i]) {
			_insert(tl[i]->k, tl[i]->v);
		}
		for(int i=0;i<ocap;i++) if(tl[i]) {
			delete tl[i];
		}
		free(tl);
	}

	~HashMap() {
		_free();
	}

	//----------------------以上是内部函数------------------------------------

	void clear() {
		_free();
		_init();
	}

	bool contains(K k){
		return _find(k) >= 0;
	}

	V& operator[](const K &k) {
		int i = _find(k);
		if(i>=0) return l[i]->v;
		i = _insert(k, V()); //把这行直接放下面，g++会挂，好神奇
		return l[i]->v;
	}

	int size() {
		return _size;
	}

	//-----iterate------------------------------
	int id;
	void beginIterate() {
		id = 0;
	}
	bool hasNext() {
		while (id < _cap && !l[id]) id++;
		return id < _cap;
	}
	Entry * next() {
		id ++;
		return l[id-1];
	}
};



int _sh(const string &s){
	unsigned int h = 0;
	for(int i=0;i<(int)s.length();i++) {
		h = (h*130 + s[i]) % 10000007;
	}
	return h;
}
int _ih(const int &i) {
	return i;
}


void process_word(char *ch) {
	bool all_large = true;
	for(int i=0;ch[i];i++) {
		if(ch[i]>='A' && ch[i]<='Z') ;
		else all_large = false;
	}
	if(!all_large) {
		for(int i=0;ch[i];i++) {
			if(ch[i]>='A' && ch[i]<='Z') {
				ch[i] += ('a'-'A');
			}
		}
	}
}

char buf[1000000], word[1000000];

HashMap<string, int, _sh> word_id;
#define MIN_OCCURENCE 60

HashMap<int, int, _ih>* cnt_map[MAX_WORD_NUMBER];

long long co_oc[MAX_WORD_NUMBER], co_oc_s;

//找出有意义的单词，在corpus文件中出现的次数不少于MIN_OCCURENCE次
void find_meaning_word(const char *corpus) {
	printf("finding meaning words\n");
	HashMap<string, int, _sh> counter(5000000);
	FILE *fr = fopen(corpus, "r");
	int len, line = 0;
	while(fgets(buf, 1000000, fr)) {
		char *p = buf;
		while(sscanf(p, " %s%n", word, &len) != EOF) {
			p += len;
			process_word(word);
			counter[word] ++;
		}
		line++;
		if(line%10000==0) {
			printf("line = %d\n", line);
		}
	}
	int i = 0;
	for(counter.beginIterate();counter.hasNext();){
		HashMap<string, int, _sh>::Entry *e = counter.next();
		if(e->v >= MIN_OCCURENCE) {
			cnt_map[i] = new HashMap<int, int, _ih>();
			word_id[e->k] = i++;
		}
	}
	printf("tot %d words, whose occurence time >= %d\n", i, MIN_OCCURENCE);
}

#define MAX_LOG 10000000
double lg[MAX_LOG];
void calc_lg() {
	for(int i=1;i<MAX_LOG;i++) {
		lg[i] = log((double)i);
	}
}

long long pos[MAX_WORD_NUMBER];

int main(){
	FILE *fr = fopen("plain_txt", "r");
	FILE *fw = fopen("out.txt", "w");
	
	char corpus[20] = "plain_txt";
	
	clock_t t = clock(), ttt = t;
	find_meaning_word(corpus);
	fprintf(fw, "find meaning words, time used = %f s\n", (clock()-t)/(double)CLOCKS_PER_SEC);
	
	//t = clock();

	int len, line = 0, max_size = 0;	
	long long word_tot = 0;
	while(fgets(buf, 1000000, fr)) {
		vector<int> vs;
		char *p = buf;
		while(sscanf(p, " %s%n", word, &len) != EOF) {
			word_tot ++;
			p += len;
			process_word(word);
			string s = word;
			if(!word_id.contains(s)) {
				vs.push_back(-1);
			} else {
				vs.push_back(word_id[s]); 
			}
		}
		for(int i=0;i<(int)vs.size();i++) if(vs[i]>=0) {
			for(int off = -3;off<=3;off++) if(off!=0) {
				int j = i + off;
				if(j>=0 && j<(int)vs.size() && vs[j]>=0) {
					(*cnt_map[vs[i]])[vs[j]] ++;
				}
			}
			max_size = max(max_size, (int)cnt_map[vs[i]]->size());
		}
		
		line++;
		if(line%10000==0) {
			printf("line = %d, max_size = %d\n", line, max_size);
		}
		//if(line>1000000) break;
	}

	vector<pair<int, string> > v;
	for(word_id.beginIterate();word_id.hasNext();) {
		HashMap<string, int, _sh>::Entry *e = word_id.next();
		v.push_back(make_pair(cnt_map[e->v]->size(), e->k));
	}
	sort(v.begin(), v.end());
	reverse(v.begin(), v.end());
	
	HashMap<int, int, _ih> id_change;
	for(int i=0;i<(int)v.size();i++) {
		id_change[word_id[v[i].second]]= i; //根据单词的vector长度来排序，越长的编号越小
	}

	fprintf(fw, "word_tot = %lld\n", word_tot);

	FILE *fout = fopen("_matrix", "wb");

	co_oc_s = 0;
	for(int i=0;i<(int)v.size();i++) {
		co_oc[i] = 0;
		for(cnt_map[i]->beginIterate();cnt_map[i]->hasNext();) {
			HashMap<int, int, _ih>::Entry *e = cnt_map[i]->next();
			co_oc[i] += e->v;
		}
		co_oc_s += co_oc[i];
	}

	calc_lg();
	double tt = log((double)co_oc_s);
	for(int i=0;i<(int)v.size();i++) {
		fprintf(fw, "%s %d\n", v[i].second.c_str(), v[i].first);
		vector<pair<int, float> > vif;  //<new id, mutual info>

		int id = word_id[v[i].second];
		int num = 0;
		float sum = 0.0;
		for(cnt_map[id]->beginIterate();cnt_map[id]->hasNext();) {
			HashMap<int, int, _ih>::Entry *e = cnt_map[id]->next();
			int tid = id_change[e->k]; //根据原来的word id得到排列后的word的id
		
//********************这里不是很懂******************
			double val = tt;
			if(e->v < MAX_LOG) val += lg[e->v];
			else val += log((double)e->v);
			if(co_oc[id] < MAX_LOG) val -= lg[co_oc[id]];
			else val -= log((double)co_oc[id]);
			if(co_oc[e->k] < MAX_LOG) val -= lg[co_oc[e->k]];
			else val -= log((double)co_oc[e->k]);

			if(val <= 0) {
				continue; //只考虑大于0的
			}
			
			num ++;
			sum += val;
			
			vif.push_back(make_pair(tid, val));
		}
		pos[i] = ftell(fout);
		fwrite(&num, sizeof(int), 1, fout);
		fwrite(&sum, sizeof(float), 1, fout);
		sort(vif.begin(), vif.end());
		for(int j=0;j<(int)vif.size();j++) {
			fwrite(&vif[j].first, sizeof(int), 1, fout);
			fwrite(&vif[j].second, sizeof(float), 1, fout);
		}
		if(i%1000 == 0) {
			printf("i = %d\n", i);
		}
	}

	FILE *fp = fopen("_words", "w");
	for(int i=0;i<(int)v.size();i++) {
		string wd = v[i].second;
		fprintf(fp, "%d %s %lld %d\n", i, wd.c_str(), pos[i], v[i].first);
	}
	fclose(fp);
	fclose(fout);

	fprintf(fw, "tot time: %f s\n", (clock()-ttt)/(double)CLOCKS_PER_SEC);
}





/*

//将一个50G的文件变成一个15G的纯文本文件
#include <vector>
#include <map>
#include <set>
#include <deque>
#include <stack>
#include <bitset>
#include <algorithm>
#include <sstream>
#include <iostream>
#include <cstdio>
#include <queue>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <string>
#include <cassert>

using namespace std;

char ch[555555];
char out[555555];
void process(char *ch, char *out) {
	char *p = out;
	int cnt = 0;
	for(int i=0;ch[i];i++) {
		if(ch[i]=='<') {
			cnt++;
			continue;
		}
		if(ch[i]=='>') {
			cnt--;
			continue;
		}
		if(cnt) continue;
		if((ch[i]>='A' && ch[i]<='Z') || (ch[i]>='a' && ch[i]<='z')) {
			*p = ch[i];
			p++;
		} else {
			*p = ' ';
			p++;
		}
	}
	*p = '\0';
}

int main() {
	FILE *f1;
	freopen_s(&f1, "I:/noname", "r", stdin);
	FILE *fp;
	fopen_s(&fp, "I:/plain_txt", "w");
	for(int i=0;fgets(ch, 555555, f1);i++) {
		if(i%10000==0) {
			printf("line = %d\n", i);
		}
		if(ch[0]=='*') continue;
		int len = strlen(ch);
		if(len<=1) continue;
		if(ch[len-1] == '\n') len--;
		if(ch[len-1] == '.') {
			process(ch, out);
			fprintf(fp, "%s\n", out);
		}
	}
	fclose(fp);
}


*/
