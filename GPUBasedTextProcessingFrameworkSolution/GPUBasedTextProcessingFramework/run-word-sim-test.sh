#!/bin/bash
for word_num in 30000 50000 100000
do
    today=$(date -d now +%Y-%m-%d-%H:%M:%S)
    out_dir="/mnt/sdb1/dd/gpu/data/wordsim-out/"$word_num"-"$today
    mkdir -p $out_dir
    log_file=$out_dir"/result.txt"
    touch $log_file
    run_code="./GPU-exe -word_sim -win_size 5 -top_words ${word_num} -corpus /mnt/sdb1/dd/gpu/data/wiki_plain_txt -cpu -gpu -output_dir ${out_dir}  -max_doc 20000000"
    echo $run_code
    $run_code >$log_file
done
