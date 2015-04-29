#include <dirent.h>
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <string>
#include <map>

using namespace std;


struct Data {
    int type; //cpu 0   gpu 1
    int doc_num;
    int block, thread, method;
    double cpu_t, gpu_t;


    bool operator<(const Data &d) const {
        if (type != d.type) {
            return type < d.type;
        }
        if (doc_num != d.doc_num) {
            return doc_num < d.doc_num;
        }
        if (block != d.block) {
            return block < d.block;
        }
        if (thread != d.thread) {
            return thread < d.thread;
        } 
        if (method != d.method) {
            return method< d.method;
        }
        return 0;
    }
};

void getTime(char *file_name, Data *data) {
    static char buf[5555];
	FILE *fp = fopen(file_name, "r");
            while (fgets(buf, 5555, fp)) {
                if (strstr(buf, "speed up")) {
                    puts(buf);
                    istringstream sin(buf);
                    string tmp;
                    while (sin >> tmp) {
                        if (tmp == "cpu_time") break;
                    }
                    sin >> tmp;
                    sin >> data->cpu_t;
                    while (sin >> tmp) {
                        if (tmp == "gpu_time") break;
                    }
                    sin >> tmp;
                    sin >> data->gpu_t;
                }
            }
}

typedef pair<int, int> PII;

int main() {
    char ddd[555] = "./data/docdup-out/";
    DIR *dir = opendir(ddd);
    dirent *file;
    Data data[5555];
    int d_cnt = 0;
    while ((file = readdir(dir)) != NULL) {
                char sss[555] = "";
                strcat(sss, ddd);
                strcat(sss, file->d_name);
                printf("%s ", file->d_name);
                if (strstr(file->d_name, "cpu")) {
                    data[d_cnt].type = 0;
                    sscanf(file->d_name, "cpu-%d%*s", &data[d_cnt].doc_num);
                } else {
                    data[d_cnt].type = 1;
                    Data &tmp = data[d_cnt];
                    sscanf(file->d_name, "gpu-%d-%d-%d-%d%*s", &tmp.doc_num, &tmp.block, &tmp.thread, &tmp.method);
                }

                getTime(sss, &data[d_cnt]);
                d_cnt ++;
    }
    sort(data, data + d_cnt);
    for (int i = 0; i < d_cnt; ++ i) {
        if (data[i].type == 0) {
            printf("cpu %6d %lfs\n", data[i].doc_num, data[i].cpu_t);
        } else {
            map<PII, double> mp[5];
            int j;
            for (j = i; j < d_cnt && data[j].doc_num == data[i].doc_num; ++ j) {
                PII p(data[j].block, data[j].thread);
                int m = data[j].method;
                if (mp[m].find(p) == mp[m].end()) {
                    mp[m][p] = data[j].gpu_t;
                } else {
                    mp[m][p] = min(data[j].gpu_t, mp[m][p]);
                }
            }
            //printf("gpu %6d, (%3d, %3d, %8d), %lfs\n", data[i].doc_num, data[i].block, data[i].thread, data[i].method, data[i].gpu_t);
            i = j - 1;
            for (int j = 1; j < 4; j ++) {
                if (j != 1 && j!= 3) {
                    continue;
                }
                printf("%d, method = %d\n", data[i].doc_num, j);
				double mi = 1e9;
                for (map<PII, double>::iterator it = mp[j].begin(); it != mp[j].end(); ++ it) {
                    printf("{%d, %d, %lf},\n", it->first.first, it->first.second, it->second);
                	if (it->second < mi) {
						mi = it->second;
					}
				}
				printf("mi = %lf\n", mi);
            }
        }
    }
    return 0;
}
