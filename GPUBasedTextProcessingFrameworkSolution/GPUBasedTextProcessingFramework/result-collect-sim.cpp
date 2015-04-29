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
    int top_words;
    int block, thread, pair_limit;
    double cpu_t, gpu_t;

    bool operator<(const Data &d) const {
        if (type != d.type) {
            return type < d.type;
        }
        if (top_words != d.top_words) {
            return top_words < d.top_words;
        }
        if (block != d.block) {
            return block < d.block;
        }
        if (thread != d.thread) {
            return thread < d.thread;
        } 
        if (pair_limit != d.pair_limit) {
            return pair_limit< d.pair_limit;
        }
        return 0;
    }
};

void getTime(char *file_name, Data *data) {
    DIR* dir = opendir(file_name);
    dirent *file;
    static char buf[5555];
    while (file = readdir(dir)) {
        if (strcmp(file->d_name, "result.txt") == 0) {
            char kkk[555] = "";
            strcat(strcat(strcat(kkk, file_name), "/"), file->d_name);
            FILE *fp = fopen(kkk, "r");
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
    }
}

int main() {
    char ddd[555] = "/mnt/sdb1/dd/gpu/data/wordsim-out/";
    DIR *dir = opendir(ddd);
    dirent *file;
    Data data[5555];
    int d_cnt = 0;
    while ((file = readdir(dir)) != NULL) {
        if (file->d_type == DT_DIR) {
            if (strstr(file->d_name, "cpu") ||
                    strstr(file->d_name, "gpu")) {
                char sss[555] = "";
                strcat(sss, ddd);
                strcat(sss, file->d_name);
                printf("%s ", file->d_name);
                if (strstr(file->d_name, "cpu")) {
                    data[d_cnt].type = 0;
                    sscanf(file->d_name, "%d%*s", &data[d_cnt].top_words);
                } else {
                    data[d_cnt].type = 1;
                    Data &tmp = data[d_cnt];
                    sscanf(file->d_name, "%d-%d-%d-%d%*s", &tmp.top_words, &tmp.block, &tmp.thread, &tmp.pair_limit);
                }

                getTime(sss, &data[d_cnt]);
                d_cnt ++;
            }
        }
    }
    sort(data, data + d_cnt);
    for (int i = 0; i < d_cnt; ++ i) {
        if (data[i].type == 0) {
            printf("cpu %6d %lfs\n", data[i].top_words, data[i].cpu_t);
        } else {

            printf("gpu %6d\n", data[i].top_words);
            int j;
            map<pair<int, int>, double> mp;
            for (j = i; j < d_cnt && data[j].top_words == data[i].top_words; ++ j) {
                printf("{%d, %d, %lf}, \n", data[j].block, data[j].thread, /*data[j].pair_limit,*/ data[j].gpu_t);
                int b = data[j].block;
                int t = data[j].thread;
                double gg = data[j].gpu_t;
                if (mp.find(make_pair(b, t)) == mp.end()) {
                    mp[make_pair(b, t)] = gg;
                } else {
                    mp[make_pair(b, t)] = min(gg, mp[make_pair(b, t)]);
                }
            }
            i = j - 1;
            printf("mapped\n");
            for (map<pair<int, int>, double>::iterator it = mp.begin(); it != mp.end(); ++ it) {
                printf("{%d, %d, %lf}, \n", it->first.first, it->first.second, it->second);
            }
        }
    }
    return 0;
}
