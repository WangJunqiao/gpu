#!/bin/bash
for doc_num in 1000 3000 5000 10000 20000 100000
do
    today=$(date -d now +%Y-%m-%d-%H:%M:%S)
    out_dir="/mnt/sdb1/dd/gpu/data/docdup-method1-out/"$doc_num"-"$today
    mkdir -p $out_dir
    log_file=$out_dir"/result.txt"
    touch $log_file
    run_code="./GPU-exe -doc_dup -files_dir /mnt/sdb1/dd/codeforce-code -cpu -gpu -max_doc ${doc_num}"
    echo $run_code
    $run_code >$log_file
done
