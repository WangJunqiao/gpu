#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("exe command");
        exit(0);
    }
    printf("wait to execute command: %s\n", argv[1]);
    bool in_test = false;
    for (int i = 1; i < argc; i ++) {
        if (strcmp(argv[i], "-test") == 0) {
            in_test = true;
        }
    }
    while (true) {
        FILE *fp = popen("nvidia-smi", "r");     
        char buf[5555];
        bool running = true;
        while (fgets(buf, 5555, fp)) {
            if (strstr(buf, "Running") != NULL ||
                    strstr(buf, "running") != NULL) {
                running = false;
            }
        }
        if (!running || in_test) {
            break;
        }
        sleep(60); //sleep for 1 minute
    }
   
    system(argv[1]);
    return 0;
}

