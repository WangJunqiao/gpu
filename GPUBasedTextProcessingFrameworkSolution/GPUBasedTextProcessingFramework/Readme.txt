This project can be compiled and run in Linux.
Use command(make) to generate GPU-exe. Use command(make clean) to clean the intermediates.

Sample command to run word similarity calculation:
./GPU-exe -word_sim -win_size 5 -top_words 10000 -corpus /mnt/sdb1/dd/gpu/data/wiki_plain_txt -cpu -gpu -output_dir /mnt/sdb1/dd/gpu/data/wordsim-out -log /mnt/sdb1/dd/gpu/data/log.txt -max_doc 500000

Sample command to run document duplication detection:
./GPU-exe -doc_dup -files_dir /mnt/sdb1/dd/codeforce-code -cpu -gpu -max_doc 1000 -log /mnt/sdb1/dd/gpu/data/docdup-out/log.txt -set_param 128 64 1

Sample command to generate idf file:
./GPU-exe -doc_clustering -idf_file /mnt/sdb1/dd/gpu/data/idf_file -calc_idf 20000 /mnt/sdb1/dd/gpu/data/wiki_plain_txt 6000000

Sample command to run document clustering:
./GPU-exe -doc_clustering -idf_file /mnt/sdb1/dd/gpu/data/idf_file -corpus /mnt/sdb1/dd/gpu/data/wiki_plain_txt -cpu -gpu -doc_num 200 -centroids 30 -log /mnt/sdb1/dd/gpu/data/log.txt

