This project can be compiled and run in Linux.
Use command(make) to generate GPU-exe. Use command(make clean) to clean the intermediates.

Sample command to run word similarity calculation:
./GPU-exe -word_sim -win_size 5 -top_words 10000 -corpus /mnt/sdb1/dd/gpu/data/wiki_plain_txt -cpu -gpu -output_dir /mnt/sdb1/dd/gpu/data/wordsim-out -log /mnt/sdb1/dd/gpu/data/log.txt -max_doc 500000


