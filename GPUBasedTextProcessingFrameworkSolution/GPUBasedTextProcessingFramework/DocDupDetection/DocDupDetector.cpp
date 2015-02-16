#include "DocDupDetector.h"

#include <string.h>

int DocDupDetector::edit_dist(const char *A,const char*B){ 
	int del_cost=1;
	int ins_cost=1;
	int ch_cost=3;
	int len1=strlen(A);
	int len2=strlen(B);
	static int dp[2][100000]; //maximum document length 100K
	int temp,i,j;
	for(int j=0;j<=len2;j++) dp[0][j] = j * ins_cost;
	int now = 0;
	for(i=0;i<len1;i++){
		dp[!now][0] = (i+1)*ins_cost;
		for(j=1;j<=len2;j++){
			dp[!now][j] = 1000000;
			if(A[i] == B[j-1]) dp[!now][j] = min(dp[!now][j], dp[now][j-1]);
			dp[!now][j] = min(dp[!now][j], dp[now][j-1] + ch_cost); //change
			dp[!now][j] = min(dp[!now][j], min(dp[now][j], dp[!now][j-1])+1); //insert delete
		}
		now = !now;
	}
	return dp[now][len2];
}

int DocDupDetector::score(const char *doc_a, const char *doc_b) {
	int ed = edit_dist(doc_a, doc_b);
	int ave_len = (strlen(doc_a), strlen(doc_b)) / 2.0;
	return ed > ave_len ? 0 : 100*(ave_len-ed) / ave_len;
}