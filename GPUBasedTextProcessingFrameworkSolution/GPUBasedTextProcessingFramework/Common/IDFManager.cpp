#include "IDFManager.h"
#include <string.h>

void IDFManager::calc_idf(DocumentSource *doc_src, const char* out_idf_file) {
	if(!doc_src->openSource()){
		LOG(logger, "%s", "cannot open document source!");
		return;
	}
	
	int total_doc_num = 0;
	map<string, int> docnum_per_word;
	set<string> s_stop_words;
	int stop_word_index = 0;
	while (_stop_words[stop_word_index]){s_stop_words.insert(_stop_words[stop_word_index]), stop_word_index++;}
	LOG(logger, "%s", "begin to calculate idf:");
	while(doc_src->hasNext()) {
		map<string, bool> word_is_in_doc;
		string s  = doc_src->getNextDocument();
		if (s == "hello")continue;
		char c[1000];
		int len, t_len = 0;
		while(sscanf(s.c_str() + t_len, "%s%n", c, &len) != -1){
			t_len += len;
			_clean_word(c);
			stem_it(c);
			if(strcmp(c, "")==0 || s_stop_words.find(c) != s_stop_words.end() ||
				word_is_in_doc[c] != 0)continue;
			docnum_per_word[c]++;
			word_is_in_doc[c] = true;
		}
		total_doc_num++;
		if (total_doc_num % 1000 == 0)LOG(logger, "processed %d documents.", total_doc_num);
	}
	LOG(logger, "%s", "documents processed finished.");
	map<string, int>::iterator it;
	int count = 0;
	FILE *t_fp = fopen(out_idf_file, "w");
	assert(t_fp != NULL);
	LOG(logger, "%s", "begin to save word_idf:");
	for (it = docnum_per_word.begin(); it != docnum_per_word.end(); it++) {
		double d = log(static_cast<double>(total_doc_num)/(docnum_per_word[it->first]+1));
		fprintf(t_fp, "%s\t%.4lf\n", (it->first).c_str(), d);
		word_idf.push_back(make_pair(it->first, d));
		count++;
		if (count % 10000 == 0)LOG(logger, "save %d words.", count);
	}
	fclose(t_fp);
}

void IDFManager::calc_idf(DocumentSource *doc_src, const int n, const char* out_idf_file) {
	if(!doc_src->openSource()){
		LOG(logger, "%s", "cannot open document source!");
		return;
	}

	int total_doc_num = 0, threshold = 1;
	unordered_map<string, int> docnum_per_word;
	map<string, int> word_count;
	set<string> s_stop_words;
	int stop_word_index = 0;
	while (_stop_words[stop_word_index]){s_stop_words.insert(_stop_words[stop_word_index]), stop_word_index++;}
	LOG(logger, "%s", "begin to calculate idf:");
	while(doc_src->hasNext()) {
		map<string, bool> word_is_in_doc;
		string s  = doc_src->getNextDocument();
		char c[1000];
		int len, t_len = 0;
		while(sscanf(s.c_str() + t_len, "%s%n", c, &len) != -1){
			t_len += len;
			_clean_word(c);
			stem_it(c);
			if(strcmp(c, "")==0 || s_stop_words.find(c) != s_stop_words.end() ||
				word_is_in_doc[c] != 0)continue;
			word_count[c]++;
			docnum_per_word[c]++;
			word_is_in_doc[c] = true;
		}
		total_doc_num++;
		if (total_doc_num == threshold) {
			LOG(logger, "processed %d documents.", total_doc_num);
			threshold *= 2;
		}
	}
	if (n < word_count.size()) {
		map<string, int>::iterator it;
		vector<pair<int, string> > count_word;
		for (it = word_count.begin(); it != word_count.end(); it++){
			count_word.push_back(make_pair(it->second, it->first));
		}
		sort(count_word.begin(), count_word.end());
		reverse(count_word.begin(), count_word.end());
		count_word.resize(n);
		int count = 0;
		LOG(logger, "%s", "begin to save word_idf:");
		FILE *fp = fopen(out_idf_file, "w");
		vector<pair<int, string> >::iterator v_it;
		unordered_map<string, int>::iterator un_it;
		for (v_it = count_word.begin(); v_it != count_word.end(); v_it++) {
			un_it = docnum_per_word.find(v_it->second);
			double d = log(static_cast<double>(total_doc_num)/(un_it->second + 1));
			fprintf(fp, "%s\t%.4lf\n", (v_it->second).c_str(), d);
			word_idf.push_back(make_pair(v_it->second, d));
			count++;
			if (count % 1000 == 0)LOG(logger, "save %d words.", count);
			if (count >= n)break;
		}
		fclose(fp);
	}
	else {
		int count = 0;
		FILE *fp = fopen(out_idf_file, "w");
		LOG(logger, "%s", "begin to save word_idf:");
		unordered_map<string, int>::iterator un_it;
		vector<pair<double, string> > idf_word;
		for (un_it = docnum_per_word.begin(); un_it != docnum_per_word.end(); un_it++) {
			double d = log(static_cast<double>(total_doc_num)/(docnum_per_word[un_it->first]+1));
			idf_word.push_back(make_pair(d, un_it->first));
			count++;
			if (count % 10000 == 0)LOG(logger, "save %d words.", count);
		}
		sort(idf_word.begin(), idf_word.end());
		reverse(idf_word.begin(), idf_word.end());
		for (int i = 0, size = idf_word.size(); i < size; i++) {
			fprintf(fp, "%s\t%.4lf\n", (idf_word[i].second).c_str(), idf_word[i].first);
		}
		fclose(fp);
	}
	LOG(logger, "%s", "documents processed finished.");
}

void IDFManager::calc_tfidf(DocumentSource *doc_src, const char *vocabulary_file, 
	const char* out_tf_idf_file, CALC_TYPE calc_type, int n) {
	if(!doc_src->openSource()){
		LOG(logger, "%s", "cannot open document source!");
		return;
	}
	if (calc_type == HAS_NO_IDF_FILE) {
		_calc_tfidf_without_idf(doc_src, vocabulary_file, out_tf_idf_file, n);
	}
	else {
		_calc_tfidf_with_idf(doc_src, vocabulary_file, out_tf_idf_file, n);
	}
}

void IDFManager::load_idf(const char* file_name) {
	assert(file_name != NULL);
	FILE *fp = fopen(file_name, "r");
	assert(fp != NULL);
	word_idf.clear();
	char c[1000]; double d;
	while (fscanf(fp, "%s%lf", c, &d) != EOF) {
		word_idf.push_back(make_pair(c, d));
	}
	fclose(fp);
}

int IDFManager::get_word_id(const char *word) {
	assert(word != NULL);
	vector<pair<string, double> >::iterator it;
	it = lower_bound(word_idf.begin(), word_idf.end(), make_pair(string(word), MIN_INF));
	if (it == word_idf.end()) return -1;
	else 
		return it - word_idf.begin() + 1;
}

double IDFManager::get_word_idf(const int& word_id) {
	assert(word_id < word_idf.size() && word_id >= 0);
	return word_idf[word_id].second;
}

double IDFManager::get_word_idf(const char *word) {
	vector<pair<string, double> >::iterator it;
	it = lower_bound(word_idf.begin(), word_idf.end(), make_pair(string(word), MIN_INF));
	if (it == word_idf.end()) return 0;
	else return it->second;
}

int IDFManager::get_word_num() {
	return word_idf.size();
}

void IDFManager::_calc_tfidf_with_idf(DocumentSource *doc_src, const char *word_idf_file, const char* out_tf_idf_file, int n) {
	FILE *fp1 = fopen(out_tf_idf_file, "w");
	FILE *fp2 = fopen(word_idf_file, "r");
	int total_doc_num = 0;
	//read the word and idf value
	unordered_map<string, double> word_idf_map;  //word and the corresponding idf value
	char word[1000]; double idf;
	while(fscanf(fp2, "%s%lf", word, &idf) != EOF) {
		word_idf_map[word] = idf;
	}
	//process the stop words
	set<string> stop_words;
	int stop_word_index = 0;
	while (_stop_words[stop_word_index]){stop_words.insert(_stop_words[stop_word_index]), stop_word_index++;}

	vector<map<string, int> > words_per_doc;  //word count of a certain document
	LOG(logger, "%s", "begin to calculate tfidf: ");
	while(doc_src->hasNext()) {
		string s  = doc_src->getNextDocument();
		char c[1000];
		map<string, int> word_count;
		int len, t_len = 0;
		while(sscanf(s.c_str() + t_len, "%s%n", c, &len) != -1){
			t_len += len;
			_clean_word(c);
			stem_it(c);
			if(strcmp(c, "")==0 || stop_words.find(c) != stop_words.end())continue; //words in the stop list
			if (word_idf_map.find(c) != word_idf_map.end())          //only process words in the vocabulary
				word_count[c]++;
		}
		words_per_doc.push_back(word_count);
		total_doc_num++;
		if (total_doc_num % 1000 == 0)LOG(logger, "processed %d documents.", total_doc_num);
	}
	LOG(logger, "%s", "documents processed finished.");
	fprintf(fp1, "%d\t%d\n", total_doc_num, (int)word_idf_map.size());
	map<string, int>::iterator it;
	unordered_map<string, double>::iterator it_d;
	vector<pair<string, double> >::iterator v_it;
	vector<pair<string, double> > v_word_idf(word_idf_map.begin(), word_idf_map.end());
	sort(v_word_idf.begin(), v_word_idf.end());
	LOG(logger, "%s", "begine to save tfidf value:");
	for (int i = 0; i < words_per_doc.size(); i++) {
		for (it = words_per_doc[i].begin(); it != words_per_doc[i].end(); it++) {
			it_d = word_idf_map.find(it->first);
			v_it = lower_bound(v_word_idf.begin(), v_word_idf.end(), make_pair(it_d->first, it_d->second));
			int word_id = v_it - v_word_idf.begin(); //word_id is the index of word in the word_idf file, start from 0.
			double t_idf = it_d->second;
			double tf_idf = static_cast<double>(it->second)*t_idf;
			fprintf(fp1, "%d\t%d\t%.4lf\n", i, word_id, tf_idf);
		}
		if (i % 1000 == 0)LOG(logger, "saved %d words.", i);
	}
	LOG(logger, "%s", "save finished.");

	fclose(fp1);
	fclose(fp2);
}

void IDFManager::_calc_tfidf_without_idf(DocumentSource *doc_src, const char *out_vocabulary_file, 
	const char* out_tf_idf_file, int n) {
	FILE *fp1 = fopen(out_tf_idf_file, "w");
	FILE *fp2 = fopen(out_vocabulary_file, "w");
	int total_doc_num = 0;
	map<string, int> docnum_per_word;  //the document number of a word contains
	vector<map<string, int> > words_per_doc;  //word count of a certain document
	map<string, int> word_count; //count the word frequency, we will only choose the top n word according to word count
	vector<string> vocabulary;
	set<string> s_stop_words;
	int stop_word_index = 0;
	while (_stop_words[stop_word_index]){s_stop_words.insert(_stop_words[stop_word_index]), stop_word_index++;}
	LOG(logger, "%s", "begin to calculate tfidf: ");
	while(doc_src->hasNext()) {
		map<string, bool> word_is_in_doc;
		map<string, int> word_count_per_doc;
		string s  = doc_src->getNextDocument();
		char c[1000];
		int len, t_len = 0;
		while(sscanf(s.c_str() + t_len, "%s%n", c, &len) != -1){
			t_len += len;
			_clean_word(c);
			stem_it(c);
			if(strcmp(c, "")==0 || s_stop_words.find(c) != s_stop_words.end())continue;
			word_count[c]++;
			word_count_per_doc[c]++;
			if(word_is_in_doc[c] != 0)break;
			docnum_per_word[c]++;
			word_is_in_doc[c] = true;
		}
		words_per_doc.push_back(word_count_per_doc);
		total_doc_num++;
		if (total_doc_num % 1000 == 0)LOG(logger, "processed %d documents.", total_doc_num);
	}
	vector<pair<int, string> > count_word;
	map<string, int>::iterator it;
	for (it = word_count.begin(); it != word_count.end(); it++)
		count_word.push_back(make_pair(it->second, it->first));
	word_count.clear();
	sort(count_word.begin(), count_word.end());
	reverse(count_word.begin(), count_word.end());
	if (n < count_word.size())count_word.resize(n);
	for (int i = 0, size = count_word.size(); i < size; i++)vocabulary.push_back(count_word[i].second);
	count_word.clear();
	sort(vocabulary.begin(), vocabulary.end());
	map<string, double> word_idf;
	for (it = docnum_per_word.begin(); it != docnum_per_word.end(); it++) {
		if (lower_bound(vocabulary.begin(), vocabulary.end(), it->first) != vocabulary.end()) {
			double d = log(static_cast<double>(total_doc_num)/(docnum_per_word[it->first]+1));
			word_idf[it->first] = d;
		}
	}
	LOG(logger, "%s", "begine to save tfidf value:");
	fprintf(fp1, "%d\t%d\n", total_doc_num, (int)vocabulary.size()); //save the doc num and word num to the head of tf-idf file
	for (int i = 0; i < total_doc_num; i++) {
		for (it = words_per_doc[i].begin(); it != words_per_doc[i].end(); it++) {
			vector<string>::iterator vec_it = lower_bound(vocabulary.begin(), vocabulary.end(), it->first);
			if (vec_it != vocabulary.end()) {
				int wordid =  vec_it - vocabulary.begin() + 1;
				double d = static_cast<double>(words_per_doc[i][it->first])*word_idf[it->first];
				fprintf(fp1, "%d\t%d\t%.4lf\n", i, wordid, d);
			}
		}
		if (i % 100 == 0)LOG(logger, "saved %d documents.", i);
		if (i == total_doc_num - 1)LOG(logger, "saved %d documents.", i);
	}
	LOG(logger, "%s", "tfidf save finished.");
	int ii, size;
	for (ii = 0, size = vocabulary.size(); ii < size; ii++){
		fprintf(fp2, "%s\n", vocabulary[ii].c_str());
		if (ii % 1000 == 0)LOG(logger, "saved %d words.", ii);
	}
	LOG(logger, "saved %d words.", ii);
	LOG(logger, "%s", "vocabulary save finished.");

	fclose(fp1);
	fclose(fp2);
}

void IDFManager::_clean_word(char *c) {
	int len = strlen(c);
	int i = 0;
	for (int j = 0; j < len; j++) {
		if(c[j] >= 'a' && c[j] <= 'z')c[i++] = c[j];
		else if (c[j] >= 'A' && c[j] <= 'Z')c[i++] = c[j] + ('a' - 'A');
	}
	c[i] = '\0';
}
