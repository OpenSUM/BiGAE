#!/bin/bash
#SBATCH
#

#preprocess
#after downloading
#export CLASSPATH=/path/to/stanford-corenlp-full-2017-06-09/stanford-corenlp-3.8.0.jar
export CLASSPATH=~/tmp/stanford-corenlp-full-2018-10-05/stanford-corenlp-3.9.2.jar

source activate py36
#RAW_PATH TOKENIZED_PATH JSON_PATH BERT_DATA_PATH MODEL_PATH
#python preprocess_presumm.py -mode tokenize -raw_path ../../data/cnndm/stories -save_path ../tk_cnndm
#python preprocess_presumm.py -mode format_to_lines -raw_path ../tk_cnndm -save_path ../new_cnndm_json/new_cnndm_json -n_cpus 1 -use_bert_basic_tokenizer false -map_path ../urls
python preprocess_presumm.py -mode format_to_bert -raw_path ../nyt/nyt_origin_data -save_path ../nyt/nyt_pre_result -lower -n_cpus 1 -log_file ./preprocess_nyt.log

