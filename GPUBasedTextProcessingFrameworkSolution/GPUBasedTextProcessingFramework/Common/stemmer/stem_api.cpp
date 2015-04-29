/* This is a simple program which uses libstemmer to provide a command
 * line interface for stemming using any of the algorithms provided.
 */
#include <set>
#include <string>
#include <stdio.h>
#include <stdlib.h> /* for malloc, free */
#include <string.h> /* for memmove */
#include <ctype.h>  /* for isupper, tolower */

#include "libstemmer.h"
#include "stem_api.h"
using namespace std;

static struct sb_stemmer *stemmer = sb_stemmer_new("english", NULL);

void stem_it(char *word) {
	int len = strlen(word);
	for(int i=0; i < len; i++) {
		if(word[i]>='A' && word[i]<='Z') word[i] = word[i] + ('a'-'A');
	}
	const sb_symbol * stemmed = sb_stemmer_stem(stemmer, (sb_symbol*)word, len);
	strcpy(word, (const char*)stemmed);
}


void run_test() {
	char words[3][20] = {
		"hElpful", 
		"carEful", 
		"" };

	for(int i=0;i<3;i++) {
		stem_it(words[i]);
		puts(words[i]);
	}
}


