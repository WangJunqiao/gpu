#!/bin/bash
    root_dir="/mnt/sdb1/dd/gpu/data/docdup-out"
    
#$1 doc_num
run_cpu() {
    today=$(date -d now +%Y-%m-%d-%H:%M:%S)
    mkdir -p $root_dir
    log_file=$root_dir"/cpu-${1}-${today}-log.txt"
    touch $log_file
    run_code="./GPU-exe -doc_dup -files_dir /mnt/sdb1/dd/codeforce-code -cpu -max_doc ${doc_num} -log ${log_file}"
    echo $run_code
    $run_code
}

for doc_num in 1000 2000 5000 10000 20000
do
    for x in 1 2; do
        run_cpu $doc_num
    done
done
