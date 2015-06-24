#!/bin/bash

root_dir="./data/wordsim-out"

common_cmd="./GPU-exe -word_sim -win_size 5 -corpus ./data/wiki_plain_txt -max_doc 20000000 -root_dir ${root_dir} "

# $1 = word_num
run_cpu(){
	today=$(date -d now +%Y-%m-%d-%H:%M:%S)
    out_dir=$root_dir"/"$1"-cpu-"$today
    mkdir -p $out_dir
    log_file=$out_dir"/result.txt"
    touch $log_file
    mi_matrix_file=$root_dir"/${1}-5-mutual_info_matrix"
	run_code=$common_cmd" -top_words ${1}  -cpu -output_dir ${out_dir} -log ${log_file}"
	if test -e $mi_matrix_file; then
		run_code=$run_code" -no_cal_mi"
    fi
    echo $run_code
    $run_code
}


for word_num in 5000 10000 20000 50000 100000
do
    run_cpu $word_num
done
