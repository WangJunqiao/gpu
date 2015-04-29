#!/bin/bash
root_dir="/mnt/sdb1/dd/gpu/data/docdup-out"
    

#$1 doc_num $2 blocks $3 threads $4 method
run_gpu() {
    today=$(date -d now +%Y-%m-%d-%H:%M:%S)

    cur_dir=$(pwd)
    cd $root_dir
    eee=$(ls | grep "gpu-"$1"-"$2"-"$3"-"$4)
    cd $cur_dir
    [ "${eee}" != "" ] && { echo "Has already calculated!"; return; }

    mkdir -p $root_dir
    log_file=$root_dir"/gpu-${1}-${2}-${3}-${4}-${today}-log.txt"
    touch $log_file
    run_code="./GPU-exe -doc_dup -files_dir /mnt/sdb1/dd/codeforce-code -gpu -max_doc ${doc_num} -log ${log_file} -set_param ${2} ${3} ${4}"
    echo $run_code
    $run_code
}

for doc_num in 2000 #5000 10000 20000
do
    for ((blocks=32;blocks<=512;blocks+=32))
    do
        for ((thread=8;thread<=96;thread+=8)) 
        do
            for method in 1
            do
                run_gpu $doc_num $blocks $thread $method
            done
        done
    done
done


for doc_num in 2000 #5000 10000 20000
do
    for ((blocks=32;blocks<=416;blocks+=32))
    do
        for ((thread=16;thread<=160;thread+=16)) 
        do
            for method in 3
            do
                run_gpu $doc_num $blocks $thread $method
            done
        done
    done
done

for doc_num in 1000 2000 5000 10000 20000
do
    for ((blocks=96;blocks<=128;blocks+=32))
    do
        for ((thread=64;thread<=96;thread+=16)) 
        do
            for method in 1 3
            do
                run_gpu $doc_num $blocks $thread $method
            done
        done
    done
done
