#!/bin/bash

root_dir="/mnt/sdb1/dd/gpu/data/wordsim-out"

common_cmd="./GPU-exe -word_sim -win_size 5 -corpus /mnt/sdb1/dd/gpu/data/wiki_plain_txt -max_doc 20000000 -root_dir ${root_dir} "

# $1 = word_num
run_cpu() {
	today=$(date -d now +%Y-%m-%d-%H:%M:%S)
    out_dir=$root_dir"/"$1"-cpu-"$today
    mkdir -p $out_dir
    log_file=$out_dir"/result.txt"
    touch $log_file
    mi_matrix_file=$root_dir"/${1}-5-mutual_info_matrix"
	run_code=$common_cmd" -top_words ${1}  -cpu -output_dir ${out_dir} "
	if test -e $mi_matrix_file then
		$run_code=$run_code" -no_cal_mi"
    echo $run_code
    $run_code >$log_file
}

# $1 = word_num
# $2 = blocks
# $3 = threads
# $4 = pair_limit
run_gpu() {
	today=$(date -d now +%Y-%m-%d-%H:%M:%S)
    out_dir=$root_dir"/"$1"-"$2"-"$3"-"$4"-gpu-"$today
    mkdir -p $out_dir
    log_file=$out_dir"/result.txt"
    touch $log_file
    mi_matrix_file=$root_dir"/${1}-5-mutual_info_matrix"
	run_code=$common_cmd" -top_words ${1}  -gpu -output_dir ${out_dir} -set_param ${2} ${3} ${4}"
	if test -e $mi_matrix_file then
		$run_code=$run_code" -no_cal_mi"
    echo $run_code
    $run_code >$log_file
}

for word_num in 10000 20000 50000 100000
do
    run_cpu $word_num
	for blocks in 128 256 512
	do
		for threads in 128 256
		do 
			for pair_limit in 30000000 50000000
			do
				run_gpu $word_num $blocks $threads $pair_limit
			done
		done
	done
done
